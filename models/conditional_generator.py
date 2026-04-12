"""
Conditional generator for diffusion models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gc
from diffusers import UNet1DModel
from tqdm import tqdm
from typing import Optional, Callable


class GradientHUNet(nn.Module):
    """Q-model: Gradient of H-function using UNet"""

    def __init__(self, in_channels: int = 4, out_channels: int = 4, sample_size: int = 64):
        super().__init__()
        self.unet = UNet1DModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            block_out_channels=(64, 128, 512),
            layers_per_block=1,
            down_block_types=("DownBlock1D", "DownBlock1D", "DownBlock1D"),
            up_block_types=("UpBlock1D", "UpBlock1D", "UpBlock1D"),
        )

    def forward(self, x, t):
        return self.unet(x, t).sample


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
        constraint_mode : str = "hard",
        beta : float = 1.0,
        in_channels: int = 5,
        sample_size: int = 64,
    ):
        self.score_model = score_model
        self.h_model = h_model
        self.diffusion_coeff_fn = diffusion_coeff_fn
        self.drift_coeff_fn = drift_coeff_fn
        self.make_vp_std_grid_fn = make_vp_std_grid_fn
        self.b_min = b_min
        self.b_max = b_max
        self.device = device

        self.q_model = None
        self.constraint_mode = constraint_mode
        self.beta = beta
        self.in_channels = in_channels
        self.sample_size = sample_size

    def train_q_model(
        self,
        t_grid: torch.Tensor,
        y_grid: torch.Tensor,
        in_channels: int = 4,
        out_channels: int = 4,
        sample_size: int = 64,
        n_epochs: int = 500,
        learning_rate: float = 1e-4,
    ) -> None:
        """
        Train Q-model (gradient of H)

        Args:
            t_grid: Time grid
            y_grid: Trajectory grid
            in_channels: Input channels
            out_channels: Output channels
            sample_size: Sample size
            n_epochs: Number of epochs
            learning_rate: Learning rate
        """
        self.in_channels = in_channels
        self.sample_size = sample_size
        self.q_model = GradientHUNet(
            in_channels=in_channels, out_channels=out_channels, sample_size=sample_size
        ).to(self.device)

        optimizer = optim.Adam(self.q_model.parameters(), lr=learning_rate)

        print("Training Q-model...")
        for epoch in range(n_epochs):
            loss = self._covariation_loss(t_grid, y_grid)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss = {loss.item():.6f}")

        print("Q-model training complete!")

    def _covariation_loss(
        self, t_grid: torch.Tensor, y_grid: torch.Tensor
    ) -> torch.Tensor:
        """Compute covariation loss for Q-model"""
        num_steps, batch_size, x_dim, seq_len = y_grid.shape

        idx = torch.randint(0, num_steps - 1, (batch_size,), device=y_grid.device)

        t = t_grid[idx, torch.arange(batch_size)]
        t_next = t_grid[idx + 1, torch.arange(batch_size)]
        y = y_grid[idx, torch.arange(batch_size)]
        y_next = y_grid[idx + 1, torch.arange(batch_size)]

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

        Returns:
            all_samples: Generated samples (num_samples, channels, seq_len)
        """
        num_full_batches = num_samples // batch_size
        remainder = num_samples % batch_size

        all_samples = []

        for i in range(num_full_batches):
            print(f"Sampling batch {i+1}/{num_full_batches} ...")
            samples = self._sample_batch(
                batch_size, num_steps, stoch, eta, use_q_model, eps
            )
            all_samples.append(samples.cpu())
            del samples
            torch.cuda.empty_cache()
            gc.collect()

        if remainder > 0:
            print(f"Sampling remainder batch of {remainder} ...")
            samples = self._sample_batch(
                remainder, num_steps, stoch, eta, use_q_model, eps
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
    ) -> torch.Tensor:
        """Sample a single batch"""
        init_x = torch.randn(batch_size, self.in_channels, self.sample_size, device=self.device)
        x = init_x

        time_steps = self.make_vp_std_grid_fn(
            self.b_min, self.b_max, num_steps, eps=eps, device=self.device
        )

        for i in tqdm(range(len(time_steps) - 1)):
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

            # Condition term
            h_val = self.h_model(x, batch_time_step)

            if use_q_model and self.q_model is not None:
                grad_h = self.q_model(x, batch_time_step)
                ratio = grad_h / (h_val.view(-1, 1, 1) + 1e-3)
            else:
                with torch.enable_grad():
                    x.requires_grad_(True)
                    h_val_autograd = self.h_model(x, batch_time_step)
                    grad_h = torch.autograd.grad(h_val_autograd.sum(), x)[0]
                ratio = grad_h / (h_val_autograd.view(-1, 1, 1) + 1e-3)
                x = x.detach()
                del h_val_autograd

            if self.constraint_mode == "hard":
                drift = drift + (1 + eta) * (g_expanded**2) * ratio.clamp(-100, 100)
            elif self.constraint_mode == "soft":
                drift = drift + (g_expanded**2 / self.beta) * grad_h


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
        self, path: str, in_channels: int = 4, out_channels: int = 4, sample_size: int = 64
    ) -> None:
        """Load Q-model weights"""
        self.in_channels = in_channels
        self.sample_size = sample_size
        self.q_model = GradientHUNet(
            in_channels=in_channels, out_channels=out_channels, sample_size=sample_size
        ).to(self.device)
        self.q_model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Q-model loaded from {path}")
