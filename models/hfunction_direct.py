import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .transformer_score import GaussianFourierFeatures, SpatioTemporalBlock
from config import HFunctionConfig
from utils import block_interleaved_epoch_order

#Building the Transformer for the conditional HFunction
class HFunctionTransformerDirect(nn.Module):

    def __init__(self, n_assets, seq_len, embed_dim, n_heads, n_layers, cond_dim, dropout=0.0):
        super().__init__()
        self.input_proj = nn.Linear(1, embed_dim)   #Embedding each scalar return into a token vector
        self.day_emb = nn.Embedding(seq_len, embed_dim)      # temporal position
        self.stock_emb = nn.Embedding(n_assets, embed_dim)   # cross-sectional identity
        self.register_buffer("day_ids", torch.arange(seq_len), persistent=False)
        self.register_buffer("stock_ids", torch.arange(n_assets), persistent=False)

        self.time_embed = nn.Sequential(
            GaussianFourierFeatures(cond_dim), #Mapping the scalar into a cond_dim vector of sines and cosines (embedding the time into a vector fomr)
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        ) #Encoding the diffusion time into a vector the transformer can condition on.

        self.blocks = nn.ModuleList([
            SpatioTemporalBlock(embed_dim, n_heads, cond_dim, dropout=dropout) for _ in range(n_layers)
        ])#n_layers blocks of joint attention over all (asset, day) tokens

        self.norm = nn.LayerNorm(embed_dim)  #Applies the layer norm to the last block
        asset_reduce_dim = 64
        self.asset_reduce = nn.Linear(embed_dim, asset_reduce_dim)

        # +1 for the end-day cross-asset dispersion feature (std across assets,
        # per embedding channel) — same asset_reduce_dim width as the other
        # asset-derived terms, but collapsed over the asset axis instead of
        # flattened, so it's a single (B, asset_reduce_dim) block, not A of them.
        head_in_dim = embed_dim + 3 * n_assets * asset_reduce_dim + asset_reduce_dim
        self.head = nn.Sequential(
            nn.Linear(head_in_dim, embed_dim//2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, 1),
        ) #Unembeds the vector representation back to a single raw logit (no Sigmoid here — see forward())


    def forward(self, x, t, return_logits: bool = False):
        #Handles weridly inputed time to make it a 1D tensor
        if t.ndim == 0:
            t = t.unsqueeze(0).expand(x.shape[0])
        elif t.ndim == 2:
            t = t.squeeze(-1)

        #Pass t through the initialized MLP
        t_emb = self.time_embed(t)

        #Tokenize: each scalar return becomes a token vector
        h = self.input_proj(x.unsqueeze(-1))                 # (B, A, T, D)

        h = (h
             + self.day_emb(self.day_ids)[None, None, :, :]      # (1, 1, T, D)
             + self.stock_emb(self.stock_ids)[None, :, None, :]) # (1, A, 1, D)

        #Flatten to joint spatiotemporal tokens, day-major: (B, A, T, D) -> (B, T*A, D)
        B, A, T, D = h.shape
        h = h.permute(0, 2, 1, 3).reshape(B, T * A, D)

        for block in self.blocks:
            h = block(h, t_emb)

        #Back to (B, A, T, D) for the start/end readout
        h = h.reshape(B, T, A, D).permute(0, 2, 1, 3)

        h = self.norm(h)                                # (B, A, T, D)
        h_mean = h.mean(dim=(1, 2))                     # (B, D) global window summary

        # Asset-resolved start/end/change features: reduce per-asset embed dim,
        # then flatten (B, A, asset_reduce_dim) -> (B, A*asset_reduce_dim) so each
        # asset keeps its own slice instead of being averaged into the others.
        z_start = self.asset_reduce(h[:, :, 0, :])    # (B, A, asset_reduce_dim)
        z_end   = self.asset_reduce(h[:, :, -1, :])   # (B, A, asset_reduce_dim)

        h_start_assets  = z_start.flatten(start_dim=1)
        h_end_assets    = z_end.flatten(start_dim=1)
        h_change_assets = (z_end - z_start).flatten(start_dim=1)

        # Cross-asset dispersion at the end day: std across the asset axis, per
        # embedding channel. h_mean/h_end_assets summarize the average asset-day
        # representation; this summarizes how SPREAD OUT the assets are from each
        # other, which the mean/per-asset-flatten terms don't directly expose
        # (e.g. "half up half down" vs "all flat" can average to the same h_mean).
        h_end_std = z_end.std(dim=1, unbiased=False)   # (B, asset_reduce_dim)

        h_pooled = torch.cat(
            [h_mean, h_start_assets, h_end_assets, h_change_assets, h_end_std], dim=-1
        )
        logits = self.head(h_pooled)
        #return_logits=True feeds BCEWithLogitsLoss directly (numerically stable with pos_weight);
        #default keeps existing probability-output behavior for sampling/generation call sites.
        if return_logits:
            return logits
        return torch.sigmoid(logits)

    
"""
Wraps the network and handles the following:
    1. Forward Noising using a VP-SDE
    2. Computing labels B
    3. Tranining using the HFunctionTransformerDirect
    4. Save/load
"""
class HFunctionDirectTrainer:

    def __init__ (self, cfg: HFunctionConfig, b_min: float, b_max: float):
        self.cfg = cfg
        self.b_min = b_min
        self.b_max = b_max
        self.device = cfg.device

        self.model = HFunctionTransformerDirect(
            n_assets=cfg.asset_dim,
            seq_len=cfg.time_steps,
            embed_dim=cfg.embed_dim,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            cond_dim=cfg.cond_dim,
            dropout=cfg.dropout,
        ).to(cfg.device)

    def _marginal_mean(self, t: torch.Tensor) -> torch.Tensor:
        integral = self.b_min * t + 0.5 * (self.b_max - self.b_min) * t ** 2
        return torch.exp(-0.5 * integral)
    
    def _marginal_std(self, t: torch.Tensor) -> torch.Tensor:
        integral = self.b_min * t + 0.5 * (self.b_max - self.b_min) * t ** 2
        return torch.sqrt(1 - torch.exp(-integral))
    
    def _forward_noise(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alpha = self._marginal_mean(t)[:, None, None]
        sigma = self._marginal_std(t)[:, None, None]
        eps = torch.randn_like(x)

        return alpha * x + sigma * eps

    def _compute_labels(self, Z_start: torch.Tensor, Z_end: torch.Tensor) -> torch.Tensor:
        thr = self.cfg.event_threshold

        if self.cfg.constraint_mode == "hard":
            if self.cfg.event_type == "abs_change":
                return (torch.abs(Z_end - Z_start) >= thr).float()
            elif self.cfg.event_type == "absval":
                return (Z_end.abs() >= thr).float()
            elif self.cfg.event_type == "upper_change":
                return ((Z_end - Z_start) >= thr).float()
            elif self.cfg.event_type == "lower_change":
                return ((Z_end - Z_start) <= -thr).float()
            elif self.cfg.event_type == "start_upper":
                return (Z_start >= thr).float()
            else:
                raise NotImplementedError(
                    f"event_type={self.cfg.event_type!r} not supported here; only 'abs_change', "
                    "'absval', 'upper_change', 'lower_change', and 'start_upper' are computable from "
                    "Z_start/Z_end alone ('sum' needs the full window, which get_z_windows() does not provide)."
                )

        elif self.cfg.constraint_mode == "soft":
            # graded sigmoid labels centered at the threshold, so every window carries
            # signal about how event-like it is instead of a hard 0/1 edge learned only
            # from the rare positives; label >= 0.5 <=> the hard event condition
            s = self.cfg.reward_sharpness
            if self.cfg.event_type == "abs_change":
                return torch.sigmoid(s * (torch.abs(Z_end - Z_start) - thr))
            elif self.cfg.event_type == "absval":
                return torch.sigmoid(s * (Z_end.abs() - thr))
            elif self.cfg.event_type == "upper_change":
                return torch.sigmoid(s * ((Z_end - Z_start) - thr))
            elif self.cfg.event_type == "lower_change":
                return torch.sigmoid(s * (-(Z_end - Z_start) - thr))
            elif self.cfg.event_type == "start_upper":
                return torch.sigmoid(s * (Z_start - thr))
            else:
                raise NotImplementedError(
                    f"event_type={self.cfg.event_type!r} not supported here; only 'abs_change', "
                    "'absval', 'upper_change', 'lower_change', and 'start_upper' are computable from "
                    "Z_start/Z_end alone ('sum' needs the full window, which get_z_windows() does not provide)."
                )

        else:
            raise NotImplementedError(
                f"constraint_mode={self.cfg.constraint_mode!r} not supported; use 'hard' or 'soft'."
            )
    
    @staticmethod
    def _episode_weights(B_labels: torch.Tensor, end_dates) -> torch.Tensor:
        """Per-window loss weight w_j for episode_reweight: 1/sqrt(m_j) for
        positive-label windows inside a run of length m_j, 1.0 for negatives.

        An "episode" is a maximal run of consecutive positive-label windows on
        the TRADING calendar (a genuine multi-day gap in the rows, e.g. missing
        macro data, breaks the run). This downweights the loss contribution
        of a single persistent macro event that spans many overlapping
        windows, so the BCE isn't dominated by however many episodes happened
        to occur rather than independent event evidence.
        """
        dates = pd.DatetimeIndex(end_dates)
        is_pos = (B_labels >= 0.5).cpu().numpy()
        # a new episode starts at index 0, or wherever the label flips to
        # positive from negative, or at a genuine gap in the rows. Adjacent
        # rows are consecutive TRADING days, so the tolerance has to absorb
        # weekends/holidays — requiring an exact 1-calendar-day step would
        # break every run across a Friday->Monday boundary and collapse
        # multi-day events into singletons.
        day_gap = np.ones(len(dates), dtype=bool)
        if len(dates) > 1:
            day_gap[1:] = (dates[1:] - dates[:-1]).days > 5
        new_episode = is_pos & (~np.r_[False, is_pos[:-1]] | day_gap)
        episode_id = np.cumsum(new_episode) - 1  # -1 for non-positive rows (unused)

        w = np.ones(len(dates), dtype=np.float64)
        if is_pos.any():
            pos_episode_id = episode_id[is_pos]
            # m_j per positive row: episode length, looked up via each row's episode id
            _, inverse, counts = np.unique(pos_episode_id, return_inverse=True, return_counts=True)
            w[is_pos] = 1.0 / np.sqrt(counts[inverse])

        return torch.tensor(w, dtype=torch.float32)

    def train(
            self,
            X_train: torch.Tensor,
            Z_start: torch.Tensor,
            Z_end: torch.Tensor,
            use_wandb: bool = False,
            end_dates=None,
            loss_csv_path: str = "ckpt_new/h_losses.csv",
    ) -> None:

        X_train = X_train.to(self.device)
        Z_start = Z_start.to(self.device)
        Z_end = Z_end.to(self.device)

        use_block_sampling = self.cfg.block_sampling
        use_episode_reweight = self.cfg.episode_reweight
        if (use_block_sampling or use_episode_reweight) and end_dates is None:
            raise ValueError(
                "cfg.block_sampling/cfg.episode_reweight require end_dates "
                "(window end dates, 1:1 aligned with X_train)."
            )

        N = X_train.shape[0]
        B_labels = self._compute_labels(Z_start, Z_end)

        # Upweight the rare positive class for training stability. This biases the
        # raw model output away from the true P(Z in S | Y_t = y) (Lemma 1) — the
        # bias is invertible (see ConditionalGenerator), so it must be corrected
        # before the model's output is used for guidance.
        n_pos = B_labels.sum().clamp(min=1)
        n_neg = (N - n_pos).clamp(min=1)
        pos_weight = (n_neg / n_pos).to(self.device)
        self.pos_weight = pos_weight.item()

        if use_episode_reweight:
            episode_weight = self._episode_weights(B_labels, end_dates).to(self.device)
            print(f"Episode reweighting: mean weight on positives = "
                  f"{episode_weight[B_labels >= 0.5].mean().item():.3f}")
        else:
            episode_weight = None

        # episode_weight multiplies the WHOLE per-element loss (1.0 on negatives,
        # 1/sqrt(m_j) on positives); pos_weight still scales the positive term
        # inside BCEWithLogitsLoss's own formula — the two stack, giving an
        # effective positive weight of pos_weight/sqrt(m_j).
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        optimizer = optim.AdamW(self.model.parameters(), lr = self.cfg.learning_rate, weight_decay= self.cfg.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode = "min", factor = self.cfg.scheduler_factor, patience= self.cfg.scheduler_patience)

        loss_records = []
        print(f"Direct H-function training | N={N} | pos_ratio={B_labels.mean():.3f} | pos_weight={pos_weight.item():.2f}")

        # Track the single best (global-minimum-loss) checkpoint seen so far, overwritten
        # in place whenever a new epoch beats it — we report/save this instead of whatever
        # the final epoch happens to land on, since the per-epoch loss curve is spiky.
        best_loss = float("inf")
        best_state = None

        block_size = self.cfg.convergence_block_size
        min_delta = self.cfg.convergence_min_delta
        block_losses = []
        prev_block_avg = None
        stagnant_blocks = 0

        for epoch in tqdm(range(self.cfg.n_epochs), desc = "HFunction-Direct Training"):
            self.model.train()

            if use_block_sampling:
                perm = block_interleaved_epoch_order(end_dates, device=self.device)
            else:
                perm = torch.randperm(N, device=self.device)

            loss_sum = acc_sum = pos_sum = 0.0
            n_batches = 0

            for start in range(0, N, self.cfg.h_mini_batch_size):
                idx = perm[start : start + self.cfg.h_mini_batch_size]
                x_b = X_train[idx]
                b_b = B_labels[idx].unsqueeze(1)

                # Cap tau at h_t_max: beyond this, Y_tau is near-pure noise and the true
                # label is unrecoverable from it, so training there just dilutes gradient
                # signal away from the range where the task is actually learnable.
                tau = torch.rand(x_b.shape[0], device = self.device) * self.cfg.h_t_max
                y_tau = self._forward_noise(x_b, tau)

                logits = self.model(y_tau, tau, return_logits=True)
                per_elem_loss = loss_fn(logits, b_b)
                if episode_weight is not None:
                    per_elem_loss = per_elem_loss * episode_weight[idx].unsqueeze(1)
                loss = per_elem_loss.mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 1.0)  #Caps the norm of the gradients at 1.0 to prevent gradients from blowing up
                optimizer.step()

                with torch.no_grad():
                    #Computes the average accuracy after we compute the mean and turn it into a python number
                    #Note Boolean.float = 1 if True 0 if False
                    prob = torch.sigmoid(logits)
                    acc = ((prob > 0.5).float() == b_b).float().mean().item()

                loss_sum += loss.item()
                acc_sum += acc
                pos_sum += b_b.mean().item()
                n_batches += 1

            avg_loss = loss_sum / n_batches
            avg_acc = acc_sum / n_batches
            avg_pos = pos_sum / n_batches
            scheduler.step(avg_loss)

            #grabs the learning rate from the optimizer
            current_lr = optimizer.param_groups[0]["lr"]
            loss_records.append({
                "epoch": epoch,
                "loss": avg_loss,
                "accuracy": avg_acc,
                "pos_ratio": avg_pos,
                "lr": current_lr,
            })

            if epoch % 100 == 0:
                tqdm.write(
                    f"Epoch {epoch:04d} | Loss: {avg_loss:.6f} | "
                    f"Acc: {avg_acc:.4f} | LR: {current_lr:.2e}"
                )

            # Global-minimum checkpoint: overwritten whenever this epoch beats the best
            # loss seen so far. Only one state dict is ever held, not one per epoch.
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

            # Dynamic convergence check: every block_size epochs, compare this block's
            # average loss to the previous block's. Two consecutive blocks that each
            # fail to improve by at least min_delta stop training early.
            block_losses.append(avg_loss)
            if len(block_losses) == block_size:
                block_avg = sum(block_losses) / block_size
                block_losses = []

                if prev_block_avg is not None:
                    improvement = prev_block_avg - block_avg
                    if improvement < min_delta:
                        stagnant_blocks += 1
                        tqdm.write(
                            f"Epoch {epoch:04d} | block avg loss {block_avg:.6f} "
                            f"(prev {prev_block_avg:.6f}, improvement {improvement:.6f} < {min_delta}) "
                            f"| stagnant blocks: {stagnant_blocks}/2"
                        )
                        if stagnant_blocks >= 2:
                            tqdm.write(f"Converged: stopping at epoch {epoch:04d} (best loss {best_loss:.6f})")
                            break
                    else:
                        stagnant_blocks = 0

                prev_block_avg = block_avg

        # Restore the best (global-minimum-loss) weights before saving/returning —
        # the run may have continued past the best epoch before triggering the stop.
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.best_loss = best_loss

        os.makedirs(os.path.dirname(loss_csv_path) or ".", exist_ok=True)
        pd.DataFrame(loss_records).to_csv(loss_csv_path, index=False)
        print("Direct H-function training complete!")
    
    def save(self, path: str) -> None:
        #pos_weight is saved alongside the weights so inference can invert the
        #training-time bias (see ConditionalGenerator) using the exact value used here.
        torch.save({
            "state_dict": self.model.state_dict(),
            "pos_weight": getattr(self, "pos_weight", 1.0),
        }, path)
        print(f"HFunctionDirect saved to {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            self.model.load_state_dict(ckpt["state_dict"])
            self.pos_weight = ckpt.get("pos_weight", 1.0)
        else:
            # Backward-compat: older checkpoints saved a plain state_dict (no pos_weight,
            # i.e. trained unweighted).
            self.model.load_state_dict(ckpt)
            self.pos_weight = 1.0
        self.model.to(self.device)
        print(f"HFunctionDirect loaded from {path} (pos_weight={self.pos_weight:.2f})")