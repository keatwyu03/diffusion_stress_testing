from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "torch")  # must be set before `import keras`

import json
from dataclasses import asdict, dataclass, field
from typing import Optional

import keras
import numpy as np
import torch
from tsgm.models.architectures.zoo import zoo
from tsgm.models.cgan import ConditionalGAN

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_default_config
from data import DataProcessor
from utils import set_seed

@dataclass
class CGANConfig:
    # Symmetric optimizer settings, one update each per batch, no label
    # smoothing -- follows TSGM's own tests/test_cgan.py example
    # (learning_rate=0.0003 for both d_optimizer and g_optimizer) rather than
    # the asymmetric d_lr/label-smoothing/discriminator-throttling combination
    # tried earlier, which is harder to justify as a "vanilla" baseline.
    latent_dim: int = 32
    batch_size: int = 64
    n_epochs: int = 1000
    g_learning_rate: float = 3e-4
    d_learning_rate: float = 3e-4
    adam_beta_1: float = 0.5
    adam_beta_2: float = 0.999
    label_smoothing: float = 0.0
    disc_update_every: int = 1
    seed: int = 2025
    n_gen_samples: Optional[int] = None  # None -> reuse config.conditional.n_gen_samples
    architecture: str = "cgan_base_c4_l1"  # TSGM zoo key (convolutional cGAN)


def compute_tanh_scale(X_train: np.ndarray) -> float:
    return float(np.max(np.abs(X_train)))


def to_tanh_range(X: np.ndarray, scale: float) -> np.ndarray:
    return (X / scale).astype(np.float32)


def from_tanh_range(X: torch.Tensor, scale: float) -> torch.Tensor:
    return X * scale


def event_mask_from_z(Z_start: torch.Tensor, Z_end: torch.Tensor, event_type: str, threshold: float) -> torch.Tensor:
    if event_type == "abs_change":
        return (Z_end - Z_start).abs() >= threshold
    if event_type == "absval":
        return Z_end.abs() >= threshold
    if event_type == "upper_change":
        return (Z_end - Z_start) >= threshold
    if event_type == "lower_change":
        return (Z_end - Z_start) <= -threshold
    if event_type == "start_upper":
        return Z_start >= threshold
    raise NotImplementedError(
        f"event_type={event_type!r} not supported; only 'abs_change', 'absval', "
        "'upper_change', 'lower_change', and 'start_upper' are implemented."
    )


def build_labels(data_processor: DataProcessor, config, split: str) -> tuple[np.ndarray, np.ndarray]:
    X = data_processor.X_train if split == "train" else data_processor.X_test
    if split == "train":
        Z_start, Z_end, valid_idx = data_processor.get_z_windows_train_aligned()
    else:
        Z_start, Z_end, valid_idx = data_processor.get_z_windows_test()

    event_type = config.hfunction.event_type
    threshold = config.hfunction.event_threshold  # already converted to raw units by caller
    event_valid = event_mask_from_z(Z_start, Z_end, event_type, threshold)

    mask = torch.zeros(X.shape[0], dtype=torch.bool)
    mask[valid_idx] = True
    B = torch.zeros(X.shape[0], dtype=torch.long)
    B[valid_idx] = event_valid.long()

    return B.numpy(), mask.numpy()


class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, labels: np.ndarray, batch_size: int, n_batches: int, seed: int):
        self.pos_idx = np.flatnonzero(labels == 1)
        self.neg_idx = np.flatnonzero(labels == 0)
        if len(self.pos_idx) == 0 or len(self.neg_idx) == 0:
            raise ValueError("BalancedBatchSampler requires at least one window of each class")
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        half = self.batch_size // 2
        for _ in range(self.n_batches):
            pos = self.rng.choice(self.pos_idx, size=half, replace=True)
            neg = self.rng.choice(self.neg_idx, size=self.batch_size - half, replace=True)
            batch = np.concatenate([pos, neg])
            self.rng.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        return self.n_batches

def build_cgan(cfg: CGANConfig, seq_len: int, feat_dim: int, output_dim: int = 2) -> ConditionalGAN:
    architecture = zoo[cfg.architecture](
        seq_len=seq_len, feat_dim=feat_dim, latent_dim=cfg.latent_dim, output_dim=output_dim
    )
    cond_gan = ConditionalGAN(
        discriminator=architecture.discriminator,
        generator=architecture.generator,
        latent_dim=cfg.latent_dim,
    )
    cond_gan.compile(
        d_optimizer=keras.optimizers.Adam(
            learning_rate=cfg.d_learning_rate, beta_1=cfg.adam_beta_1, beta_2=cfg.adam_beta_2
        ),
        g_optimizer=keras.optimizers.Adam(
            learning_rate=cfg.g_learning_rate, beta_1=cfg.adam_beta_1, beta_2=cfg.adam_beta_2
        ),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=cfg.label_smoothing),
    )
    return cond_gan


def train_cgan(
    cond_gan: ConditionalGAN,
    X_train: np.ndarray,
    B_train: np.ndarray,
    cfg: CGANConfig,
) -> list[dict]:
    """Returns per-epoch [{"epoch", "g_loss", "d_loss", "minimax_value"}, ...].

    minimax_value is the joint GAN value V(D,G) = E[log D(x)] + E[log(1-D(G(z)))],
    computed as -d_loss: TSGM's d_loss is BinaryCrossentropy(desc_labels,
    predictions) over the concatenated real+fake batch with labels 1/0, which
    is exactly -V(D,G) under the standard GAN formulation (BCE loss is the
    negative of what the discriminator maximizes). No separate forward pass
    needed -- this reuses the d_loss TSGM already computes during train_step.
    """
    y = keras.utils.to_categorical(B_train, 2).astype(np.float32)
    X_tensor = torch.from_numpy(X_train)
    y_tensor = torch.from_numpy(y)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)

    n_batches_per_epoch = max(1, len(X_train) // cfg.batch_size)
    losses = []

    disc_lr = cfg.d_learning_rate
    batch_counter = 0

    for epoch in range(cfg.n_epochs):
        sampler = BalancedBatchSampler(
            B_train, batch_size=cfg.batch_size, n_batches=n_batches_per_epoch,
            seed=cfg.seed + epoch,
        )
        # Materialize this epoch's batch index groups up front so each one can
        # be re-wrapped as its own single-batch DataLoader below (needed for
        # per-batch discriminator throttling -- see comment below).
        batch_index_groups = list(sampler)

        epoch_g_losses, epoch_d_losses = [], []
        for indices in batch_index_groups:
            # disc_update_every > 1 throttles the discriminator (updates it
            # on only 1 of every disc_update_every batches, by zeroing its
            # optimizer's learning rate on the skipped batches) so it can't
            # overpower the generator as quickly -- addresses the oscillating
            # d_loss/g_loss pattern and cross-asset mode collapse observed
            # with 1:1 updates. The generator still updates every batch
            # regardless (train_step_torch always runs both, this only
            # controls whether the discriminator's update actually moves it).
            #
            # Uses fit() on a real single-batch DataLoader rather than
            # train_on_batch(): Keras's train_on_batch(x, y) packages data as
            # a 3-tuple (x, y, sample_weight) internally, but TSGM's
            # train_step_torch only handles a 2-tuple (real_ts, labels) --
            # with train_on_batch, labels silently comes through as None.
            # fit() on an actual torch DataLoader preserves the 2-tuple the
            # loader itself yields, which is the code path TSGM was written
            # for (a plain Python list of tensors was NOT enough -- Keras's
            # data adapter handles a real DataLoader differently).
            do_disc_update = (batch_counter % cfg.disc_update_every == 0)
            cond_gan.d_optimizer.learning_rate = disc_lr if do_disc_update else 0.0

            single_batch = torch.utils.data.Subset(dataset, indices)
            single_batch_loader = torch.utils.data.DataLoader(single_batch, batch_size=len(indices))
            history = cond_gan.fit(single_batch_loader, epochs=1, verbose=0)
            epoch_g_losses.append(float(history.history["g_loss"][-1]))
            epoch_d_losses.append(float(history.history["d_loss"][-1]))
            batch_counter += 1

        cond_gan.d_optimizer.learning_rate = disc_lr

        g_loss = float(np.mean(epoch_g_losses))
        d_loss = float(np.mean(epoch_d_losses))
        minimax_value = -d_loss
        losses.append({
            "epoch": epoch, "g_loss": g_loss, "d_loss": d_loss, "minimax_value": minimax_value,
        })
        if epoch % max(1, cfg.n_epochs // 20) == 0 or epoch == cfg.n_epochs - 1:
            print(f"[cGAN] epoch {epoch+1}/{cfg.n_epochs}  g_loss={g_loss:.4f}  "
                  f"d_loss={d_loss:.4f}  V(D,G)={minimax_value:.4f}")

    return losses

def generate_conditional(cond_gan: ConditionalGAN, n_samples: int, label_value: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    labels = keras.utils.to_categorical(np.full(n_samples, label_value, dtype=np.int64), 2).astype(np.float32)
    gen = cond_gan.generate(labels)
    gen_np = gen.detach().cpu().numpy() if hasattr(gen, "detach") else np.asarray(gen)
    gen_t = torch.from_numpy(gen_np).permute(0, 2, 1).contiguous()  # (N,T,A) -> (N,A,T)
    return gen_t

def run_diagnostics(gen_b1: torch.Tensor, gen_b0: torch.Tensor, seq_len: int, n_assets: int) -> dict:
    diags = {}

    diags["shape_b1"] = tuple(gen_b1.shape)
    diags["shape_b0"] = tuple(gen_b0.shape)
    diags["shape_ok"] = (gen_b1.shape[1:] == (n_assets, seq_len)) and (gen_b0.shape[1:] == (n_assets, seq_len))

    diags["finite_b1"] = bool(torch.isfinite(gen_b1).all())
    diags["finite_b0"] = bool(torch.isfinite(gen_b0).all())

    b1_np = gen_b1.numpy()
    b0_np = gen_b0.numpy()

    diags["b1_vs_b0_lastday_mean_abs_diff"] = float(
        np.abs(b1_np[:, :, -1].mean(axis=0) - b0_np[:, :, -1].mean(axis=0)).mean()
    )
    diags["b1_vs_b0_meaningfully_different"] = diags["b1_vs_b0_lastday_mean_abs_diff"] > 1e-3

    per_asset_var = b1_np.var(axis=(0, 2))  # (A,)
    diags["b1_per_asset_variance"] = per_asset_var.tolist()
    diags["min_per_asset_variance"] = float(per_asset_var.min())

    n_sub = min(200, b1_np.shape[0])
    flat = b1_np[:n_sub].reshape(n_sub, -1)
    if n_sub > 1:
        diffs = flat[:, None, :] - flat[None, :, :]
        dists = np.sqrt((diffs ** 2).sum(-1))
        iu = np.triu_indices(n_sub, k=1)
        diags["mean_pairwise_l2_distance"] = float(dists[iu].mean())
        diags["min_pairwise_l2_distance"] = float(dists[iu].min())
    else:
        diags["mean_pairwise_l2_distance"] = None
        diags["min_pairwise_l2_distance"] = None

    if n_sub > 1:
        eps = 1e-4 * np.sqrt(flat.shape[1])
        diags["fraction_near_duplicate_pairs"] = float((dists[iu] < eps).mean())
    else:
        diags["fraction_near_duplicate_pairs"] = None

    last_day = b1_np[:, :, -1]  # (N, A)
    if last_day.shape[0] > 1:
        corr = np.corrcoef(last_day, rowvar=False)
        diags["cross_asset_corr_matrix"] = corr.tolist()
    else:
        diags["cross_asset_corr_matrix"] = None

    return diags


def main(cgan_cfg: Optional[CGANConfig] = None):
    if cgan_cfg is None:
        cgan_cfg = CGANConfig()
    set_seed(cgan_cfg.seed)

    config = get_default_config()

    data_processor = DataProcessor(
        csv_path=config.data.csv_path,
        tickers=config.data.tickers,
        weekday_col=config.data.weekday_col,
        seq_len=config.data.seq_len,
        test_days=config.data.test_days,
        start_date=config.data.start_date,
        end_date=config.data.end_date,
        train_end_date=config.data.train_end_date,
        window_shift=config.data.window_shift,
        winsorize_lower=config.data.winsorize_lower,
        winsorize_upper=config.data.winsorize_upper,
        ema_span=config.data.ema_span,
        event_causal=config.data.event_causal,
        event_lag_gap=config.data.event_lag_gap,
    )
    data_processor.process_all()

    event_top_fraction = config.hfunction.event_threshold
    config.hfunction.event_threshold = data_processor.get_event_threshold_from_percentile(
        event_top_fraction, config.hfunction.event_type
    )
    print(f"Event threshold: top {event_top_fraction:.1%} -> "
          f"{config.hfunction.event_threshold:.4f} std ({config.hfunction.event_type})")

    n_assets = len(config.data.tickers)
    seq_len = config.data.seq_len

    # X_train/X_test are (N, T, A) -- already TSGM-native time-major, no transpose needed here.
    X_train = data_processor.X_train.numpy().astype(np.float32)
    X_test = data_processor.X_test.numpy().astype(np.float32)

    B_train, mask_train = build_labels(data_processor, config, split="train")
    B_test, mask_test = build_labels(data_processor, config, split="test")

    # Only train on windows with a valid macro observation (mask_train==True);
    # this subsets the existing training set, does not duplicate or alter rows.
    X_train_valid = X_train[mask_train]
    B_train_valid = B_train[mask_train]

    n_event_train = int(B_train_valid.sum())
    n_event_test = int((B_test[mask_test]).sum())
    print(f"Train set size (valid): {len(X_train_valid)}  |  events: {n_event_train} "
          f"({n_event_train / len(X_train_valid):.1%})")
    print(f"Test set size (valid): {int(mask_test.sum())}  |  events: {n_event_test}")

    tanh_scale = compute_tanh_scale(X_train_valid)
    print(f"tanh_scale (train-set max |X|): {tanh_scale:.6f} -- "
          f"generator trains on X/tanh_scale, in [-1,1]; generated output is "
          f"multiplied back by tanh_scale before use.")
    X_train_scaled = to_tanh_range(X_train_valid, tanh_scale)

    # ---- Build & train ----
    cond_gan = build_cgan(cgan_cfg, seq_len=seq_len, feat_dim=n_assets)
    losses = train_cgan(cond_gan, X_train_scaled, B_train_valid, cgan_cfg)

    # ---- Generate: TWO conditional (B=1) draws, mirroring the diffusion
    #      pipeline's generated_samples_train.pt / generated_samples_test.pt.
    #
    #      The GAN itself is the SAME trained generator in both cases (it is
    #      only ever trained on X_train_valid -- never on held-out test
    #      windows). What differs, matching main.py's ConditionalGenerator
    #      usage and PortfolioAnalyzer.analyze_samples()'s in-sample vs.
    #      out-of-sample comparison, is which real-event population each
    #      draw is meant to be compared against, and correspondingly which
    #      entry-stat (mu, sig) pool is used to destandardize it back to raw
    #      returns:
    #        - "train": entry stats drawn from IN-SAMPLE (train) event windows
    #        - "test":  entry stats drawn from OUT-OF-SAMPLE (test) event windows
    #      Each draw also uses its own independent generation seed (like
    #      main.py's two separate cond_generator.generate() calls). ----
    n_gen = cgan_cfg.n_gen_samples if cgan_cfg.n_gen_samples is not None else config.conditional.n_gen_samples

    print(f"Generating {n_gen} conditional (B=1) samples for in-sample (train) comparison...")
    gen_b1_train = from_tanh_range(generate_conditional(cond_gan, n_gen, label_value=1, seed=cgan_cfg.seed), tanh_scale)

    print(f"Generating {n_gen} conditional (B=1) samples for out-of-sample (test) comparison...")
    gen_b1_test = from_tanh_range(generate_conditional(cond_gan, n_gen, label_value=1, seed=cgan_cfg.seed + 1000), tanh_scale)

    # B=0 draws are for the conditioning diagnostic only (requirement: "generate
    # both B=0 and B=1 samples for the conditioning diagnostic, even though the
    # primary evaluation concerns B=1") -- not saved as a primary output.
    print(f"Generating {n_gen} conditional (B=0) samples for the conditioning diagnostic...")
    gen_b0_diag = from_tanh_range(generate_conditional(cond_gan, n_gen, label_value=0, seed=cgan_cfg.seed + 2000), tanh_scale)

    # Reproducibility check: same seed -> same output, WITHIN A SMALL TOLERANCE.
    # cuDNN's LSTM kernel (used by the zoo's cgan_base_c4_l1 generator) is not
    # bit-exact reproducible on GPU even with every documented determinism flag
    # set (torch.manual_seed + cuda.manual_seed_all + cudnn.deterministic=True
    # + torch.use_deterministic_algorithms(True) all verified insufficient --
    # this is a known PyTorch/cuDNN limitation for RNN kernels specifically,
    # not a seeding bug here). The tolerance is scaled by tanh_scale since
    # gen_b1_train is now in raw (post-inverse-scale) units, not [-1,1] units.
    REPRO_ATOL = 1e-2 * tanh_scale
    gen_b1_train_repeat = from_tanh_range(generate_conditional(cond_gan, n_gen, label_value=1, seed=cgan_cfg.seed), tanh_scale)
    repro_max_abs_diff = float((gen_b1_train - gen_b1_train_repeat).abs().max())
    reproducible = repro_max_abs_diff < REPRO_ATOL

    # ---- Diagnostics (train-side B=1 draw vs. the B=0 diagnostic draw) ----
    diags = run_diagnostics(gen_b1_train, gen_b0_diag, seq_len=seq_len, n_assets=n_assets)
    diags["reproducible_under_fixed_seed"] = reproducible
    diags["reproducibility_max_abs_diff"] = repro_max_abs_diff
    diags["reproducibility_tolerance"] = REPRO_ATOL
    diags["reproducibility_note"] = (
        "Not bit-exact on GPU due to cuDNN LSTM kernel nondeterminism (documented "
        "PyTorch/cuDNN limitation, not a seeding bug); checked within tolerance instead."
    )
    print("\n=== Diagnostics ===")
    for k, v in diags.items():
        if k != "cross_asset_corr_matrix":
            print(f"  {k}: {v}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gan_results")
    os.makedirs(out_dir, exist_ok=True)

    torch.save(gen_b1_train, os.path.join(out_dir, "gan_generated_samples_train.pt"))
    torch.save(gen_b1_test, os.path.join(out_dir, "gan_generated_samples_test.pt"))

    cond_gan.generator.save(os.path.join(out_dir, "cgan_generator.keras"))
    cond_gan.discriminator.save(os.path.join(out_dir, "cgan_discriminator.keras"))

    with open(os.path.join(out_dir, "cgan_losses.json"), "w") as f:
        json.dump(losses, f, indent=2)

    with open(os.path.join(out_dir, "cgan_diagnostics.json"), "w") as f:
        json.dump(diags, f, indent=2)

    run_config = {
        "cgan_config": asdict(cgan_cfg),
        "tanh_scale": tanh_scale,
        "event_type": config.hfunction.event_type,
        "event_threshold_raw": config.hfunction.event_threshold,
        "event_threshold_top_fraction": event_top_fraction,
        "seq_len": seq_len,
        "n_assets": n_assets,
        "n_train_valid": len(X_train_valid),
        "n_event_train": n_event_train,
        "n_event_test": n_event_test,
        "n_gen_samples": n_gen,
        "seed": cgan_cfg.seed,
    }
    with open(os.path.join(out_dir, "cgan_run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"\nSaved outputs to {out_dir}:")
    print("  gan_generated_samples_train.pt        (N,A,T) STANDARDIZED B=1 draw #1 (seed).")
    print("                                         Intended comparison target:")
    print("                                         X_train[mask_train & B_train==1]")
    print("  gan_generated_samples_test.pt         (N,A,T) STANDARDIZED B=1 draw #2 (seed+1000,")
    print("                                         independent of draw #1). Intended comparison")
    print("                                         target: X_test[mask_test & B_test==1]")
    print("  cgan_generator.keras, cgan_discriminator.keras       reloadable TSGM/Keras model checkpoints")
    print("  cgan_losses.json                      per-epoch g_loss/d_loss")
    print("  cgan_diagnostics.json                 shape/finiteness/diversity/correlation checks")
    print("                                         (computed on draw #1 vs. a separate,")
    print("                                         independently-seeded B=0 diagnostic-only draw)")
    print("  cgan_run_config.json                  full config + seed for reproducibility")
    print()
    print("NOTE: this script trains, generates, runs diagnostics, and saves samples only.")
    print("It does NOT run the analysis/ evaluation scripts itself and never writes to the")
    print("repo-root generated_samples_{train,test}.pt used by diffusion-mode evaluation.")
    print("To evaluate these samples, run:  python evaluation_main.py --gan")


if __name__ == "__main__":
    main()
