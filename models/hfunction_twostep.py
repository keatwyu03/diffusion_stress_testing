import torch
import torch.nn as nn
import numpy as np
import os

from config import HFunctionConfig, get_default_config
from .transformer_score import DualAxisBlock, GaussianFourierFeatures

class EllTransformer(nn.Module):
    #input -> 
    #each layer: time attention head -> norm -> asset attention head -> norm -> FFN -> norm
    #final norm -> head -> probability 
    #-> output

    def __init__(self, n_assets, seq_len, embed_dim, n_heads, n_layers):
        super().__init__()
        #(batch, assets, time, embedding dimension)
        #Batch_first = True so we have batch size first
        self.input_proj = nn.Linear(1, embed_dim)
        self.temporal_pos = nn.Parameter(torch.randn(1, 1, seq_len, embed_dim) * 0.02)
        self.asset_emb = nn.Parameter(torch.randn(1, n_assets, 1, embed_dim) * 0.02)
        
        #n_layers of each temporal attention, asset attention, and feed forward network
        self.temporal_attns = nn.ModuleList([nn.MultiheadAttention(embed_dim, n_heads, batch_first=True) for _ in range(n_layers)])
        self.asset_attns = nn.ModuleList([nn.MultiheadAttention(embed_dim, n_heads, batch_first=True) for _ in range(n_layers)])
        self.ffns = nn.ModuleList(
            [nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), 
                           nn.GELU(), 
                           nn.Linear(embed_dim * 4, embed_dim)) 
                           for _ in range(n_layers)]
                           )

        #Norms in between every head at every layer
        self.norms = nn.ModuleList(
            [nn.LayerNorm(embed_dim) for _ in range(n_layers * 3)])
        
        #Final norm is computed before we head for the probability
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, A, T = x.shape
        h = self.input_proj(x.unsqueeze(-1))
        h = h + self.temporal_pos + self.asset_emb

        for i in range(len(self.temporal_attns)):
            #Temporal attention blocks
            h_t = h.reshape(B*A, T, h.shape[-1])
            h_t, _ = self.temporal_attns[i](h_t, h_t, h_t)
            h = h + self.norms[i*3](h_t.reshape(B, A, T, -1))

            #Asset attention blocks
            h_a = h.permute(0, 2, 1, 3).reshape(B*T, A, h.shape[-1])
            h_a, _ = self.asset_attns[i](h_a, h_a, h_a)
            h = h + self.norms[i*3 + 1](h_a.reshape(B, T, A, -1).permute(0, 2, 1, 3))

            #FFN
            h = h + self.norms[i*3 + 2](self.ffns[i](h))
        
        h = self.norm(h).mean(dim=(1,2))
        return self.head(h)



class EllTrainer:
    def __init__(self, cfg: HFunctionConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = EllTransformer(
            n_assets = cfg.asset_dim,
            seq_len = cfg.time_steps,
            embed_dim = cfg.embed_dim, 
            n_heads = cfg.n_heads,
            n_layers = cfg.n_layers
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = cfg.learning_rate, weight_decay= cfg.weight_decay)
    
    def train(self, X_train, Z_start, Z_end, use_wandb = False):
        B_labels = (torch.abs(Z_end - Z_start) >= self.cfg.event_threshold).float().to(self.device)
        n_pos = B_labels.sum()
        n_neg = len(B_labels) - n_pos
        pos_weight = n_neg / n_pos
        loss_fn = nn.BCELoss(reduction = "none")
        
        X_train = X_train.to(self.device)
        for epoch in range(self.cfg.n_epochs):
            idx = torch.randperm(len(X_train))[:self.cfg.h_mini_batch_size]
            x_batch = X_train[idx]
            b_batch = B_labels[idx]
            
            self.optimizer.zero_grad()
            pred = self.model(x_batch).squeeze(-1)
            raw_loss = loss_fn(pred, b_batch)
            sample_weight = torch.where(b_batch == 1, pos_weight.expand_as(b_batch), torch.ones_like(b_batch))

            loss = (raw_loss * sample_weight).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            if epoch % 100 == 0:
                acc = ((pred > 0.5) == b_batch.bool()).float().mean()
                pos_ratio = b_batch.mean()
                print(f"Epoch {epoch} | Loss {loss.item():.4f} | Acc {acc.item():.4f} | Pos ratio {pos_ratio.item():.4f}")

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

class HFunctionTransformerTwoStep(nn.Module):
    def __init__(self, n_assets, seq_len, embed_dim, n_heads, n_layers, cond_dim):
        super().__init__()
        self.input_proj = nn.Linear(1, embed_dim)
        self.temporal_pos = nn.Parameter(torch.randn(1, 1, seq_len, embed_dim) * 0.02)
        
        self.asset_emb = nn.Parameter(torch.randn(1, n_assets, 1, embed_dim) * 0.02)
        self.time_embed = nn.Sequential(
            GaussianFourierFeatures(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )

        self.blocks = nn.ModuleList([DualAxisBlock(embed_dim, n_heads, cond_dim) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x, t):
        if t.ndim == 0:
            t = t.unsqueeze(0).expand(x.shape[0])
        elif t.ndim == 2:
            t = t.squeeze(-1)
        
        t_emb = self.time_embed(t)
        h = self.input_proj(x.unsqueeze(-1))
        
        h = h + self.temporal_pos + self.asset_emb
        for block in self.blocks:
            h = block(h, t_emb)
        
        h = self.norm(h).mean(dim = (1,2))
        return self.head(h)







class HFunctionTwoStepTrainer:
    def __init__(self, cfg: HFunctionConfig, diffusion_model, ell_model):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.diffusion_model = diffusion_model
        self.ell_model = ell_model
        self.model = HFunctionTransformerTwoStep(
            n_assets=cfg.asset_dim,
            seq_len=cfg.time_steps,
            embed_dim=cfg.embed_dim,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            cond_dim=cfg.embed_dim
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = cfg.learning_rate, weight_decay= cfg.weight_decay)
    
    def train(self, use_wandb = False):
        loss_fn = nn.MSELoss()
        for epoch in range(self.cfg.n_epochs):
            with torch.no_grad():
                paths = self.diffusion_model.sample(
                    batch_size = self.cfg.train_batch_size,
                    return_path = True
                )
                Y_T = paths[:, -1]
                ell_labels = self.ell_model(Y_T).squeeze(-1)
            
            t_idx = torch.randint(0, paths.shape[1], (self.cfg.train_batch_size,))
            Y_t = paths[torch.arange(self.cfg.train_batch_size), t_idx]
            t = t_idx.float() / (paths.shape[1] - 1)

            self.optimizer.zero_grad()
            pred = self.model(Y_t.to(self.device), t.to(self.device)).squeeze(-1)
            loss = loss_fn(pred, ell_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch} | Loss {loss.item():.4f}")

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))