"""
Dual-axis Transformer score network for financial time series diffusion models.

Architecture:
  - Per-element input projection: each scalar return → embed_dim
  - Learnable temporal positional embedding + per-asset embedding
  - Diffusion time t conditioned via Gaussian Fourier features + AdaLN
  - N dual-axis transformer blocks:
      1. Temporal self-attention  (each asset attends over its 64 time steps)
      2. Cross-asset self-attention (each time step attends over all 4 assets)
      3. Position-wise FFN
  - Output projection back to scalar per position
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ScoreOutput:
    """Wraps raw tensor to match HuggingFace Diffusers .sample interface."""
    sample: torch.Tensor


class GaussianFourierFeatures(nn.Module):
    def __init__(self, embed_dim: int, scale: float = 30.0):
        super().__init__()
        self.W = nn.Parameter(
            torch.randn(embed_dim // 2) * scale, requires_grad=False
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        proj = t * self.W[None, :] * 2 * math.pi
        return torch.cat([proj.sin(), proj.cos()], dim=-1)


class AdaLN(nn.Module):
    """
    Adaptive Layer Norm for time conditioning.
    Accepts x of any shape (..., dim) and cond of shape (batch, cond_dim).
    Broadcasts scale/shift over all intermediate dims automatically.
    """

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 2 * dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(cond).chunk(2, dim=-1)  # (batch, dim)
        for _ in range(x.ndim - 2):                      # insert intermediate dims
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        return self.norm(x) * (1 + scale) + shift


class DualAxisBlock(nn.Module):
    """
    One transformer block with two attention axes and a FFN.

    Temporal axis : (B*A, T, D) — each of the 4 assets attends over 64 time steps.
    Asset axis    : (B*T, A, D) — each time step attends over all 4 assets.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        cond_dim: int,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Temporal self-attention
        self.temporal_norm = AdaLN(dim, cond_dim)
        self.temporal_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )

        # Cross-asset self-attention
        self.asset_norm = AdaLN(dim, cond_dim)
        self.asset_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )

        # FFN
        self.ffn_norm = AdaLN(dim, cond_dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
        )

        # Residual dropout after each sub-layer (no-op when dropout=0.0, e.g. the score model)
        self.temporal_drop = nn.Dropout(dropout)
        self.asset_drop = nn.Dropout(dropout)
        self.ffn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: (B, A, T, D),  t_emb: (B, C)
        B, A, T, D = x.shape
        C = t_emb.shape[-1]

        # ── Temporal self-attention ────────────────────────────────────────
        x_t = x.reshape(B * A, T, D)
        c_t = t_emb.unsqueeze(1).expand(B, A, C).reshape(B * A, C)
        normed = self.temporal_norm(x_t, c_t)
        out, _ = self.temporal_attn(normed, normed, normed)
        x = x + self.temporal_drop(out.reshape(B, A, T, D))

        # ── Cross-asset self-attention ─────────────────────────────────────
        x_a = x.permute(0, 2, 1, 3).reshape(B * T, A, D)
        c_a = t_emb.unsqueeze(1).expand(B, T, C).reshape(B * T, C)
        normed = self.asset_norm(x_a, c_a)
        out, _ = self.asset_attn(normed, normed, normed)
        x = x + self.asset_drop(out.reshape(B, T, A, D).permute(0, 2, 1, 3))

        # ── FFN ────────────────────────────────────────────────────────────
        x_flat = x.reshape(B * A * T, D)
        c_flat = t_emb.unsqueeze(1).unsqueeze(1).expand(B, A, T, C).reshape(B * A * T, C)
        normed = self.ffn_norm(x_flat, c_flat)
        x = x + self.ffn_drop(self.ffn(normed).reshape(B, A, T, D))

        return x


class SpatioTemporalBlock(nn.Module):
    """One transformer block with JOINT attention over all (asset, day) tokens.

    Unlike DualAxisBlock, which factorizes attention into a temporal pass and a
    cross-asset pass (so cross-asset, cross-time information needs two hops),
    every token here attends to every other token directly — "asset i on day 3
    -> asset j on day 9" forms in a single hop. At A*T tokens (e.g. 40) the full
    attention matrix is tiny, so factorization buys nothing.

    Diffusion time conditions each sub-layer through AdaLN (t_emb sets the
    scale/shift of the normalization), same mechanism as DualAxisBlock.
    """

    def __init__(self, dim, n_heads, cond_dim, ff_mult=4, dropout=0.0):
        super().__init__()
        self.attn_norm = AdaLN(dim, cond_dim)
        self.attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = AdaLN(dim, cond_dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
        )
        self.attn_drop = nn.Dropout(dropout)
        self.ffn_drop = nn.Dropout(dropout)

    def forward(self, x, t_emb):
        # x: (B, N, D) flattened spatiotemporal tokens, t_emb: (B, cond_dim)
        normed = self.attn_norm(x, t_emb)
        out, _ = self.attn(normed, normed, normed)
        x = x + self.attn_drop(out)

        normed = self.ffn_norm(x, t_emb)
        x = x + self.ffn_drop(self.ffn(normed))
        return x


class FinancialTransformerScore(nn.Module):
    """
    Full dual-axis Transformer score network.

    Default params (embed_dim=128, n_layers=6) give ~2M parameters,
    comparable to the original UNet.
    """

    def __init__(
        self,
        n_assets: int = 4,
        seq_len: int = 64,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 6,
        cond_dim: int = 128,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_assets = n_assets
        self.seq_len = seq_len

        # Each scalar return value → embed_dim
        self.input_proj = nn.Linear(1, embed_dim)

        # Learned positional tables: each of the seq_len days and each of the
        # n_assets stocks gets its own trainable embedding vector, same
        # tokenization style as HFunctionTransformerDirect.
        self.day_emb = nn.Embedding(seq_len, embed_dim)      # temporal position
        self.stock_emb = nn.Embedding(n_assets, embed_dim)   # cross-sectional identity
        self.register_buffer("day_ids", torch.arange(seq_len), persistent=False)
        self.register_buffer("stock_ids", torch.arange(n_assets), persistent=False)

        # Time conditioning: Fourier features → small MLP → cond_dim
        self.time_embed = nn.Sequential(
            GaussianFourierFeatures(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.blocks = nn.ModuleList([
            SpatioTemporalBlock(embed_dim, n_heads, cond_dim, ff_mult, dropout)
            for _ in range(n_layers)
        ])

        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> ScoreOutput:
        # x: (B, A, T),  t: (B,) or scalar tensor
        if t.ndim == 0:
            t = t.unsqueeze(0).expand(x.shape[0])

        t_emb = self.time_embed(t)               # (B, cond_dim)

        h = self.input_proj(x.unsqueeze(-1))      # (B, A, T, embed_dim)
        h = (h
             + self.day_emb(self.day_ids)[None, None, :, :]      # (1, 1, T, D)
             + self.stock_emb(self.stock_ids)[None, :, None, :]) # (1, A, 1, D)

        # Flatten to joint spatiotemporal tokens, day-major: (B, A, T, D) -> (B, T*A, D)
        B, A, T, D = h.shape
        h = h.permute(0, 2, 1, 3).reshape(B, T * A, D)

        for block in self.blocks:
            h = block(h, t_emb)

        # Back to (B, A, T, D) for the per-position output head
        h = h.reshape(B, T, A, D).permute(0, 2, 1, 3)

        out = self.output_proj(self.output_norm(h)).squeeze(-1)  # (B, A, T)
        return ScoreOutput(sample=out)
