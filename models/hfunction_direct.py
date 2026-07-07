import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .transformer_score import GaussianFourierFeatures, DualAxisBlock
from config import HFunctionConfig


#Building the Neural Network
class HFunctionTransformerDirect(nn.Module):

    def __init__(self, n_assets, seq_len, embed_dim, n_heads, n_layers, cond_dim):
        super().__init__()
        self.input_proj = nn.Linear(1, embed_dim)   #Embedding the cross section vector
        self.temporal_pos = nn.Parameter(torch.randn(1,1, seq_len, embed_dim) * 0.02)    #Setup the positional random vector to learn positional unique impact
        self.asset_emb = nn.Parameter(torch.randn(1, n_assets, 1, embed_dim) * 0.02)     #Setup the asset random vector to learn asset unique impact
        
        self.time_embed = nn.Sequential(
            GaussianFourierFeatures(cond_dim), #Mapping the scalar into a cond_dim vector of sines and cosines (embedding the time into a vector fomr)
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(), 
            nn.Linear(cond_dim, cond_dim)
        ) #Encoding the diffusion time into a vector the transformer can condition on. 

        self.blocks = nn.ModuleList([
            DualAxisBlock(embed_dim, n_heads, cond_dim) for _ in range(n_layers)
        ])#Sets up n_layers dual axis blocks which are feed forward NN

        self.norm = nn.LayerNorm(embed_dim)  #Applies the layer norm to the last block
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid(),
        ) #Unembeds the vector representation back to a single value

    
    def forward(self, x, t):
        #Handles weridly inputed time to make it a 1D tensor
        if t.ndim == 0:
            t = t.unsqueeze(0).expand(x.shape[0])
        elif t.ndim == 2:
            t = t.squeeze(-1)
        
        #Pass t through the initialized MLP
        t_emb = self.time_embed(t)

        #Embedding the cross sectino
        h = self.input_proj(x.unsqueeze(-1))

        #Adds the temporal and asset based embeddings to the vectors
        h = h + self.temporal_pos + self.asset_emb

        #Passes each vector throguh through each DualAxisBlock 
        for block in self.blocks:
            h = block(h, t_emb)

        #Passes it through the AdaLN and then decompresses it
        h = self.norm(h).mean(dim=(1,2))
        return self.head(h)

    
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
        return (torch.abs(Z_end - Z_start) >= self.cfg.event_threshold).float()
    
    def train(
            self,
            X_train: torch.Tensor,
            Z_start: torch.Tensor,
            Z_end: torch.Tensor,
            use_wandb: bool = False,
    ) -> None:
        
        X_train = X_train.to(self.device)
        Z_start = Z_start.to(self.device)
        Z_end = Z_end.to(self.device)

        N = X_train.shape[0]
        B_labels = self._compute_labels(Z_start, Z_end)

        # Plain (unweighted) BCE: the direct classifier's population minimizer must be
        # P(Z in S | Y_t = y) exactly (Lemma 1). Upweighting positives shifts that
        # minimizer in logit space, biasing the propagated likelihood used for guidance.
        loss_fn = nn.BCELoss(reduction='mean')
        optimizer = optim.AdamW(self.model.parameters(), lr = self.cfg.learning_rate, weight_decay= self.cfg.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode = "min", factor = self.cfg.scheduler_factor, patience= self.cfg.scheduler_patience)

        loss_records = []
        print(f"Direct H-function training | N={N} | pos_ratio={B_labels.mean():.3f}")

        for epoch in tqdm(range(self.cfg.n_epochs), desc = "HFunction-Direct Training"):
            self.model.train()

            idx = torch.randint(0, N, (self.cfg.h_mini_batch_size,), device = self.device)
            x_b = X_train[idx]
            b_b = B_labels[idx].unsqueeze(1)

            tau = torch.rand(self.cfg.h_mini_batch_size, device = self.device)
            y_tau = self._forward_noise(x_b, tau)

            prob = self.model(y_tau, tau)
            loss = loss_fn(prob, b_b)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 1.0)  #Caps the norm of the gradients at 1.0 to prevent gradients from blowing up 
            optimizer.step()
            scheduler.step(loss)

            with torch.no_grad():
                #Computes the average accuracy after we compute the mean and turn it into a python number
                #Note Boolean.float = 1 if True 0 if False
                acc = ((prob > 0.5).float() == b_b).float().mean().item()

            
            #grabs the learning rate from the optimizer
            current_lr = optimizer.param_groups[0]["lr"]
            loss_records.append({
                "epoch": epoch,
                "loss": loss.item(), 
                "accuracy": acc,
                "pos_ratio": b_b.mean().item(),
                "lr": current_lr,
            })

            if epoch % 100 == 0:
                tqdm.write(
                    f"Epoch {epoch:04d} | Loss: {loss.item():.6f} | "
                    f"Acc: {acc:.4f} | LR: {current_lr:.2e}"
                )
        
        os.makedirs("ckpt_new", exist_ok=True)
        pd.DataFrame(loss_records).to_csv("ckpt_new/h_losses.csv", index=False)
        print("Direct H-function training complete!")
    
    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)     #Returns a dictionary of the model's learned parameters
        print(f"HFunctionDirect saved to {path}")

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        print(f"HFunctionDirect loaded from {path}")