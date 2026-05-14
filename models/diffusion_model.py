"""
Diffusion model for financial time series generation
"""
import torch
import torch.nn as nn
import functools
import numpy as np
from diffusers import UNet1DModel
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Tuple, Optional, Callable
from tqdm import tqdm


class DiffusionModel:
    """Variance Preserving (VP) Diffusion Model"""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        sample_size: int = 64,
        layers_per_block: int = 2,
        block_out_channels: Tuple[int, int, int] = (64, 256, 512),
        b_min: float = 0.1,
        b_max: float = 3.25,
        device: str = "cuda",
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sample_size = sample_size
        self.b_min = b_min
        self.b_max = b_max
        self.device = device

        # Create UNet model
        self.model = UNet1DModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=("DownBlock1D", "DownBlock1D", "DownBlock1D"),
            up_block_types=("UpBlock1D", "UpBlock1D", "UpBlock1D"),
            time_embedding_type="fourier",
            freq_shift=6,
        ).to(device)

        # Create VP diffusion functions
        self.marginal_prob_mean_fn = functools.partial(
            self.marginal_prob_mean, b_min=b_min, b_max=b_max
        )
        self.marginal_prob_std_fn = functools.partial(
            self.marginal_prob_std, b_min=b_min, b_max=b_max
        )
        self.diffusion_coeff_fn = functools.partial(
            self.diffusion_coeff, b_min=b_min, b_max=b_max
        )
        self.drift_coeff_fn = functools.partial(
            self.drift_coeff, b_min=b_min, b_max=b_max
        )

    @staticmethod
    def marginal_prob_mean(t, b_min, b_max):
        """Marginal probability mean at time t"""
        t = torch.as_tensor(t)
        integral = b_min * t + 0.5 * (b_max - b_min) * t**2
        return torch.exp(-0.5 * integral)

    @staticmethod
    def marginal_prob_std(t, b_min, b_max):
        """Marginal probability std at time t"""
        t = torch.as_tensor(t)
        integral = b_min * t + 0.5 * (b_max - b_min) * t**2
        return torch.sqrt(1 - torch.exp(-integral))

    @staticmethod
    def diffusion_coeff(t, b_min, b_max):
        """Diffusion coefficient at time t"""
        t = torch.as_tensor(t)
        return torch.sqrt(b_min + (b_max - b_min) * t)

    @staticmethod
    def drift_coeff(t, b_min, b_max):
        """Drift coefficient at time t"""
        t = torch.as_tensor(t)
        return -0.5 * (b_min + (b_max - b_min) * t)

    def loss_fn(
        self,
        x: torch.Tensor,
        marginal_prob_mean: Callable,
        marginal_prob_std: Callable,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """
        Loss function for training the score-based model

        Args:
            x: Input data (batch_size, channels, seq_len)
            marginal_prob_mean: Function to compute mean at time t
            marginal_prob_std: Function to compute std at time t
            eps: Small epsilon value for numerical stability

        Returns:
            loss: Scalar loss value
        """
        self.model.train()
        batch_size = x.shape[0]
        device = x.device

        random_t = torch.rand(batch_size, device=device) * (1.0 - eps) + eps
        z = torch.randn_like(x)
        std = marginal_prob_std(random_t)
        std_expanded = std[:, None, None]
        mean = marginal_prob_mean(random_t)
        mean_expanded = mean[:, None, None]

        perturbed_x = x * mean_expanded + z * std_expanded

        score = self.model(perturbed_x, random_t).sample

        # loss = torch.mean(torch.sum((score * std_expanded + z) ** 2, dim=(1, 2)))
        std_clamped = std_expanded.clamp(min=1e-3)
        loss = torch.mean(torch.sum((score + z / std_clamped) ** 2, dim=(1, 2)))

        return loss

    def train(
        self,
        train_data: torch.Tensor,
        batch_size: int = 256,
        n_epochs: int = 600,
        learning_rate: float = 1e-4,
        scheduler_patience: int = 50,
        scheduler_factor: float = 0.5,
        num_workers: int = 0,
        use_wandb: bool = False,
    ) -> None:
        """
        Train the diffusion model

        Args:
            train_data: Training data tensor
            batch_size: Batch size
            n_epochs: Number of epochs
            learning_rate: Learning rate
            scheduler_patience: Patience for learning rate scheduler
            scheduler_factor: Factor for learning rate reduction
            num_workers: Number of data loader workers
            use_wandb: Whether to log metrics to wandb
        """
        if use_wandb:
            import wandb

        print(f"[DEBUG] Train data shape: {train_data.shape}", flush=True)
        print(f"[DEBUG] Device: {self.device}", flush=True)
        print(f"[DEBUG] CUDA available: {torch.cuda.is_available()}", flush=True)
        print(f"[DEBUG] GPU count: {torch.cuda.device_count()}", flush=True)

        dataset = TensorDataset(train_data)
        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        print(f"[DEBUG] DataLoader created, batches: {len(data_loader)}", flush=True)

        # Disable multi-GPU for now (can cause issues)
        # if torch.cuda.device_count() > 1:
        #     self.model = nn.DataParallel(self.model)

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience
        )

        print("Starting diffusion model training...", flush=True)
        for epoch in tqdm(range(n_epochs), desc="Diffusion Training"):
            avg_loss = 0.0
            num_items = 0
            for (x,) in data_loader:
                x = x.to(self.device)

                loss = self.loss_fn(
                    x, self.marginal_prob_mean_fn, self.marginal_prob_std_fn
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                avg_loss += loss.item() * x.shape[0]
                num_items += x.shape[0]

            avg_loss /= num_items
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(avg_loss)

            # Log to wandb
            if use_wandb:
                wandb.log({
                    "diffusion/loss": avg_loss,
                    "diffusion/learning_rate": current_lr,
                    "diffusion/epoch": epoch + 1,
                })

            # Log to console periodically
            if (epoch + 1) % 50 == 0:
                tqdm.write(
                    f"Epoch [{epoch+1}/{n_epochs}]  "
                    f"Loss: {avg_loss:.6f}, LR: {current_lr:.2e}"
                )

        # Unwrap DataParallel if used
        if isinstance(self.model, nn.DataParallel):
            self.model = self.model.module

        print("Diffusion model training complete!")

    @staticmethod
    def make_vp_std_grid(
        b_min: float, b_max: float, num_steps: int, eps: float = 1e-3, device: str = "cuda"
    ) -> torch.Tensor:
        """Create adaptive time grid based on VP diffusion std"""
        Δ = b_max - b_min
        I_tot = b_min + 0.5 * Δ
        sigma_tot = torch.sqrt(1.0 - torch.exp(torch.tensor(-I_tot, device=device)))

        t_list = []
        for i in range(num_steps, -1, -1):  # i: N..0 → t: ~1..eps
            σ_i = (i / num_steps) * sigma_tot
            I_i = -torch.log(
                torch.clamp(1.0 - σ_i**2, min=1e-12)
            )  # ∫β = -log(1 - σ²)
            disc = b_min**2 + 2.0 * Δ * I_i
            t_i = (-b_min + torch.sqrt(disc)) / (Δ + 1e-12)
            t_list.append(torch.clamp(t_i, min=eps))

        return torch.stack(t_list).to(device)

    def sample(
        self,
        batch_size: int = 64,
        num_steps: int = 200,
        stoch: float = 1.0,
        eps: float = 1e-4,
        return_path: bool = False,
    ) -> torch.Tensor:
        """
        Sample from the diffusion model using Euler-Maruyama

        Args:
            batch_size: Number of samples
            num_steps: Number of sampling steps
            stoch: Stochasticity parameter (0=deterministic, 1=full stochastic)
            eps: Small epsilon for numerical stability
            return_path: Whether to return the full path

        Returns:
            samples: Generated samples
        """
        self.model.eval()

        init_x = torch.randn(batch_size, self.in_channels, self.sample_size, device=self.device)
        x = init_x

        time_steps = self.make_vp_std_grid(
            self.b_min, self.b_max, num_steps, eps=eps, device=self.device
        )

        path_t, path_x = [], []

        with torch.no_grad():
            for i in range(len(time_steps) - 1):
                time_step = time_steps[i]
                next_t = time_steps[i + 1]
                step_size = (time_step - next_t).abs()

                batch_time_step = torch.ones(batch_size, device=self.device) * time_step

                g = self.diffusion_coeff_fn(batch_time_step)
                g_expanded = g[:, None, None]
                f = self.drift_coeff_fn(batch_time_step)
                f_expanded = f[:, None, None]

                score = self.model(x, batch_time_step).sample
                adjust = (1 + stoch**2) / 2
                mean_x = (
                    x + (-f_expanded * x + adjust * (g_expanded**2) * score) * step_size
                )
                x = mean_x + stoch * torch.sqrt(step_size) * g_expanded * torch.randn_like(x)

                if return_path:
                    path_t.append(batch_time_step.clone())
                    path_x.append(x.clone())

        if return_path:
            path_t = torch.stack(path_t)
            path_x = torch.stack(path_x)
            return path_t, path_x, mean_x
        else:
            return mean_x

    def save(self, path: str) -> None:
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model weights"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        print(f"Model loaded from {path}")
