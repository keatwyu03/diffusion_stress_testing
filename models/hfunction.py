"""
H-function for conditional generation
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Tuple, Callable
from tqdm import tqdm


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps"""

    def __init__(self, embed_dim: int, scale: float = 30.0):
        super().__init__()
        self.W = nn.Parameter(
            torch.randn(embed_dim // 2) * scale, requires_grad=False
        )

    def forward(self, t):
        if t.ndim == 1:
            t = t[:, None]
        t_proj = t * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)


class HFunctionCNN(nn.Module):
    """H-function network using CNN architecture"""

    def __init__(self, asset_dim: int = 4, time_steps: int = 64, embed_dim: int = 128):
        super().__init__()
        self.asset_dim = asset_dim
        self.time_steps = time_steps

        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
        )

        self.conv_net = nn.Sequential(
            nn.Conv1d(asset_dim, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv1d(16, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv1d(64, 256, 3, padding=1),
            nn.GroupNorm(16, 256),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.fc = nn.Sequential(
            nn.Linear(256 + embed_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, t):
        if t.ndim == 1:
            t = t[:, None]
        t_emb = self.time_embed(t)
        feat = self.conv_net(x).squeeze(-1)
        out = self.fc(torch.cat([feat, t_emb], dim=1))
        return out


class HFunctionTrainer:
    """Trainer for H-function"""

    def __init__(
        self,
        asset_dim: int = 4,
        time_steps: int = 64,
        embed_dim: int = 128,
        event_asset_idx: int = 3,
        event_window: int = 10,
        event_threshold: float = -3.0,
        device: str = "cuda",
        event_type : str = "sum",
        constraint_mode : str = "hard",
        reward_sharpness : float = 10
        
    ):
        self.asset_dim = asset_dim
        self.time_steps = time_steps
        self.event_asset_idx = event_asset_idx
        self.event_window = event_window
        self.event_threshold = event_threshold
        self.device = device
        self.event_type = event_type
        self.constraint_mode = constraint_mode
        self.reward_sharpness = reward_sharpness

        # Create model
        self.model = HFunctionCNN(
            asset_dim=asset_dim, time_steps=time_steps, embed_dim=embed_dim
        ).to(device)

    def train(
        self,
        t_grid: torch.Tensor,
        y_grid: torch.Tensor,
        Y_T: torch.Tensor,
        n_epochs: int = 400,
        batch_size: int = 2**13,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        scheduler_patience: int = 50,
        scheduler_factor: float = 0.5,
        use_wandb: bool = False,
    ) -> None:
        """
        Train the H-function

        Args:
            t_grid: Time grid (num_steps, batch_size)
            y_grid: Trajectory grid (num_steps, batch_size, asset_dim, time_steps)
            Y_T: Terminal states (batch_size, asset_dim, time_steps)
            n_epochs: Number of epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            weight_decay: Weight decay
            scheduler_patience: Patience for scheduler
            scheduler_factor: Factor for scheduler
            use_wandb: Whether to log metrics to wandb
        """
        if use_wandb:
            import wandb

        B = t_grid.shape[1]

        optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience
        )
        loss_fn = nn.MSELoss()

        print("Starting H-function training...")
        for epoch in tqdm(range(n_epochs), desc="H-Function Training"):
            # Sample random time steps and trajectories
            t_idx = torch.randint(0, t_grid.shape[0], (batch_size,), device=self.device)
            b_idx = torch.randint(0, B, (batch_size,), device=self.device)

            chosen_t = t_grid[t_idx, b_idx]
            chosen_x = y_grid[t_idx, b_idx]

            # Compute event indicator from terminal state
            terminal = Y_T[b_idx]
            last_window = terminal[:, self.event_asset_idx, -self.event_window :]
            if self.event_type == "sum":
                sum_last_window = last_window.sum(dim=1)
                target = (sum_last_window <= self.event_threshold).float().unsqueeze(1)
            elif self.event_type == "change":
                diff_over_window = abs(last_window[:, -1] - last_window[:, 0])
                if self.constraint_mode == "soft":
                    target = torch.sigmoid(self.reward_sharpness * diff_over_window).unsqueeze(1)
                elif self.constraint_mode == "hard":
                    target = (diff_over_window >= self.event_threshold).float().unsqueeze(1)
            elif self.event_type == "absval":
                abs_val = last_window[:, -1].abs()
                if self.constraint_mode == "soft":
                    target = torch.sigmoid(self.reward_sharpness * abs_val).unsqueeze(1)
                elif self.constraint_mode == "hard":
                    target = (abs_val >= self.event_threshold).float().unsqueeze(1)

            # Forward pass
            pred = self.model(chosen_x, chosen_t)
            loss = loss_fn(pred, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(loss)

            # Compute metrics
            acc = ((pred > 0.5).float() == target).float().mean().item()
            pos_ratio = target.mean().item()

            # Log to wandb
            if use_wandb:
                wandb.log({
                    "hfunction/loss": loss.item(),
                    "hfunction/accuracy": acc,
                    "hfunction/pos_ratio": pos_ratio,
                    "hfunction/learning_rate": current_lr,
                    "hfunction/epoch": epoch,
                })

            # Log to console periodically
            if epoch % 100 == 0:
                tqdm.write(
                    f"Epoch {epoch:04d} | Loss: {loss.item():.6f} | "
                    f"Acc: {acc:.4f} | PosRatio: {pos_ratio:.3f}"
                )

        print("H-function training complete!")

    def save(self, path: str) -> None:
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
        print(f"H-function saved to {path}")

    def load(self, path: str) -> None:
        """Load model weights"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        print(f"H-function loaded from {path}")
