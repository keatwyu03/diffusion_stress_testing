"""
Conditional generator for diffusion models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gc
from tqdm import tqdm
from typing import Optional, Callable

from .transformer_score import FinancialTransformerScore


class GradientHUNet(nn.Module):
    """Q-model: approximates ∇H using a Transformer score network."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        sample_size: int = 64,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        cond_dim: int = 64,
    ):
        super().__init__()
        self.transformer = FinancialTransformerScore(
            n_assets=in_channels,
            seq_len=sample_size,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            cond_dim=cond_dim,
        )

    def forward(self, x, t):
        return self.transformer(x, t).sample


class ConditionalGenerator:
    """Conditional sample generator"""

    def __init__(
        self,
        score_model: nn.Module,
        h_model: nn.Module,
        diffusion_coeff_fn: Callable,
        drift_coeff_fn: Callable,
        make_vp_std_grid_fn: Callable,
        b_min: float = 0.1,
        b_max: float = 3.25,
        device: str = "cuda",
        h_t_max: float = 1.0,
        pos_weight: float = 1.0,
    ):
        self.score_model = score_model
        self.h_model = h_model
        self.diffusion_coeff_fn = diffusion_coeff_fn
        self.drift_coeff_fn = drift_coeff_fn
        self.make_vp_std_grid_fn = make_vp_std_grid_fn
        self.b_min = b_min
        self.b_max = b_max
        self.device = device
        # Guidance is only applied for tau <= h_t_max, matching the range the
        # h-function was actually trained on — beyond that, Y_tau is near-pure
        # noise and h's gradient there is unreliable, not just weak.
        self.h_t_max = h_t_max
        # h_model was trained with pos_weight upweighting the rare positive class,
        # which shifts its raw output away from the true P(Z in S | Y_t=y). This
        # inverts that shift back to the calibrated probability before it's used
        # for guidance. pos_weight=1.0 (default) makes the correction a no-op.
        self.pos_weight = pos_weight

        self.q_model = None

    def train_q_model(
        self,
        t_grid: torch.Tensor,
        y_grid: torch.Tensor,
        in_channels: int = 4,
        out_channels: int = 4,
        sample_size: int = 64,
        n_epochs: int = 500,
        learning_rate: float = 1e-4,
        mini_batch_size: int = 512,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        cond_dim: int = 64,
    ) -> None:
        """Train Q-model (gradient of H)"""
        self.q_model = GradientHUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            sample_size=sample_size,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            cond_dim=cond_dim,
        ).to(self.device)

        optimizer = optim.Adam(self.q_model.parameters(), lr=learning_rate)

        print("Training Q-model...")
        for epoch in range(n_epochs):
            loss = self._covariation_loss(t_grid, y_grid, mini_batch_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss = {loss.item():.6f}")

        print("Q-model training complete!")

    def _covariation_loss(
        self, t_grid: torch.Tensor, y_grid: torch.Tensor, mini_batch_size: int = 512
    ) -> torch.Tensor:
        """Compute covariation loss for Q-model"""
        num_steps, n_paths, x_dim, seq_len = y_grid.shape

        # Sample mini_batch_size (step, path) pairs randomly from all paths
        step_idx = torch.randint(0, num_steps - 1, (mini_batch_size,), device=y_grid.device)
        path_idx = torch.randint(0, n_paths, (mini_batch_size,), device=y_grid.device)

        t = t_grid[step_idx, path_idx]
        t_next = t_grid[step_idx + 1, path_idx]
        y = y_grid[step_idx, path_idx]
        y_next = y_grid[step_idx + 1, path_idx]

        h_t = self.h_model(y, t)
        h_next = self.h_model(y_next, t_next)
        delta_h = (h_next - h_t).squeeze(-1)

        delta_y = y_next - y

        g_val = self.diffusion_coeff_fn(t)
        delta_t = (t_next - t).abs()
        denom = g_val**2 * delta_t

        approx_grad = (delta_h[:, None, None] * delta_y) / (denom[:, None, None] + 1e-4)

        q_pred = self.q_model(y, t)

        loss = ((approx_grad - q_pred) ** 2).mean()
        return loss

    @torch.no_grad()
    def generate(
        self,
        num_samples: int,
        batch_size: int = 32,
        num_steps: int = 200,
        stoch: float = 0.3,
        eta: float = 150.0,
        use_q_model: bool = False,
        eps: float = 1e-5,
        stop_early_steps: int = 0,
    ) -> torch.Tensor:
        """
        Generate conditional samples

        Args:
            num_samples: Total number of samples to generate
            batch_size: Batch size for generation
            num_steps: Number of diffusion steps
            stoch: Stochasticity parameter
            eta: Guidance strength
            use_q_model: Whether to use Q-model
            eps: Epsilon for numerical stability
            stop_early_steps: Stop this many steps before reaching the end of the
                reverse SDE (t=eps), leaving some residual noise/diversity in the
                samples instead of fully resolving to the sharp end state.

        Returns:
            all_samples: Generated samples (num_samples, channels, seq_len)
        """
        num_full_batches = num_samples // batch_size
        remainder = num_samples % batch_size

        all_samples = []

        for i in range(num_full_batches):
            print(f"Sampling batch {i+1}/{num_full_batches} ...")
            samples = self._sample_batch(
                batch_size, num_steps, stoch, eta, use_q_model, eps, stop_early_steps
            )
            all_samples.append(samples.cpu())
            del samples
            torch.cuda.empty_cache()
            gc.collect()

        if remainder > 0:
            print(f"Sampling remainder batch of {remainder} ...")
            samples = self._sample_batch(
                remainder, num_steps, stoch, eta, use_q_model, eps, stop_early_steps
            )
            all_samples.append(samples.cpu())
            del samples
            torch.cuda.empty_cache()
            gc.collect()

        all_samples = torch.cat(all_samples, dim=0)
        print(f"Final samples shape: {all_samples.shape}")
        return all_samples

    def _sample_batch(
        self,
        batch_size: int,
        num_steps: int,
        stoch: float,
        eta: float,
        use_q_model: bool,
        eps: float,
        stop_early_steps: int = 0,
    ) -> torch.Tensor:
        """Sample a single batch"""
        self.score_model.eval()
        self.h_model.eval()
        if self.q_model is not None:
            self.q_model.eval()

        n_assets = self.score_model.n_assets
        seq_len  = self.score_model.seq_len
        init_x = torch.randn(batch_size, n_assets, seq_len, device=self.device)
        x = init_x

        time_steps = self.make_vp_std_grid_fn(
            self.b_min, self.b_max, num_steps, eps=eps, device=self.device
        )

        n_iters = max(len(time_steps) - 1 - stop_early_steps, 1)
        for i in tqdm(range(n_iters)):
            time_step = time_steps[i]
            next_t = time_steps[i + 1]
            step_size = (time_step - next_t).abs()

            batch_time_step = torch.full((batch_size,), time_step, device=self.device)
            g = self.diffusion_coeff_fn(batch_time_step)
            g_expanded = g[:, None, None]
            f = self.drift_coeff_fn(batch_time_step)
            f_expanded = f[:, None, None]

            # Base drift from score model
            score = self.score_model(x, time_step).sample
            drift = (g_expanded**2) * score

            # Guidance only applied for tau <= h_t_max — beyond that, Y_tau is near-pure
            # noise and the h-function was never trained there, so its gradient is
            # unreliable rather than just weak. Skip it entirely instead of risking a
            # spurious nudge that compounds through the rest of the trajectory.
            if time_step.item() <= self.h_t_max:
                # Using h's raw output directly (no pos_weight correction) — matching
                # the reference implementation, which doesn't correct for it either.
                if use_q_model and self.q_model is not None:
                    h_val = self.h_model(x, batch_time_step)
                    grad_h = self.q_model(x, batch_time_step)
                    ratio = grad_h / (h_val.view(-1, 1, 1) + 1e-3)
                else:
                    with torch.enable_grad():
                        x.requires_grad_(True)
                        h_val_autograd = self.h_model(x, batch_time_step)
                        grad_h = torch.autograd.grad(h_val_autograd.sum(), x)[0]
                    ratio = grad_h / (h_val_autograd.view(-1, 1, 1) + 1e-3)
                    x = x.detach()
                    del h_val_autograd, grad_h

                drift = drift + (1 + eta) * (g_expanded**2) * ratio

            # Euler-Maruyama update
            adjust = (1 + stoch**2) / 2
            mean_x = x + (-f_expanded * x + adjust * drift) * step_size
            x = mean_x + stoch * torch.sqrt(step_size) * g_expanded * torch.randn_like(x)

        return mean_x

    def save_q_model(self, path: str) -> None:
        """Save Q-model weights"""
        if self.q_model is not None:
            torch.save(self.q_model.state_dict(), path)
            print(f"Q-model saved to {path}")

    def load_q_model(
        self,
        path: str,
        in_channels: int = 4,
        out_channels: int = 4,
        sample_size: int = 64,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        cond_dim: int = 64,
    ) -> None:
        """Load Q-model weights"""
        self.q_model = GradientHUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            sample_size=sample_size,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            cond_dim=cond_dim,
        ).to(self.device)
        self.q_model.load_state_dict(torch.load(path))
        print(f"Q-model loaded from {path}")
