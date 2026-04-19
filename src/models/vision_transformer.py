"""Vision Transformer components for I-JEPA adapted for OCT data.

Adapted from the original I-JEPA repository
(https://github.com/facebookresearch/ijepa) with modifications for OCT
volume processing including slice-level encoders and predictors.

Compatible with PyTorch 1.13.1.
"""

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from src.masks.utils import apply_masks
from src.utils.tensors import trunc_normal_, repeat_interleave_batch

# ---------------------------------------------------------------------------
# Embedding dimension look-up for factory functions
# ---------------------------------------------------------------------------
VIT_EMBED_DIMS = {
    "vit_tiny": 192,
    "vit_small": 384,
    "vit_base": 768,
    "vit_large": 1024,
    "vit_huge": 1280,
}


# ===================================================================
# Positional embeddings
# ===================================================================

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """Generate 2-D sinusoidal positional embeddings.

    Args:
        embed_dim: Total embedding dimension (must be even).
        grid_size: int or (H, W) — number of patches along each spatial axis.

    Returns:
        pos_embed: np.ndarray of shape (grid_size*grid_size, embed_dim) or
            (H*W, embed_dim) when grid_size is a tuple.
    """
    if isinstance(grid_size, int):
        grid_h, grid_w = grid_size, grid_size
    else:
        grid_h, grid_w = grid_size

    grid_h_arr = np.arange(grid_h, dtype=np.float32)
    grid_w_arr = np.arange(grid_w, dtype=np.float32)
    grid = np.meshgrid(grid_w_arr, grid_h_arr)  # (W, H) ordering from meshgrid
    grid = np.stack(grid, axis=0)  # (2, H, W)
    grid = grid.reshape([2, 1, grid_h, grid_w])

    pos_embed = _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed  # (H*W, D)


def get_1d_sincos_pos_embed(embed_dim, grid_size_or_positions):
    """Generate 1-D sinusoidal positional embeddings.

    Args:
        embed_dim: Embedding dimension (must be even).
        grid_size_or_positions: Either an int (number of positions, 0..N-1) or
            a 1-D np.ndarray of position values.

    Returns:
        pos_embed: np.ndarray of shape (N, embed_dim).
    """
    if isinstance(grid_size_or_positions, (int, np.integer)):
        positions = np.arange(grid_size_or_positions, dtype=np.float32)
    else:
        positions = np.asarray(grid_size_or_positions, dtype=np.float32)

    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000.0 ** omega)  # (D/2,)

    pos = positions.reshape(-1, 1)  # (N, 1)
    out = pos * omega[None, :]  # (N, D/2)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (N, D)
    return emb


def _get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """2-D sincos pos_embed from a (2, 1, H, W) grid produced by np.meshgrid(w, h)."""
    assert embed_dim % 2 == 0
    # meshgrid(w, h) with default 'xy' indexing:
    #   grid[0]: (1, H, W) width coords  (constant along rows)
    #   grid[1]: (1, H, W) height coords (constant along cols)
    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, grid[1, 0, :, 0])  # (H, D/2)
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, grid[0, 0, 0, :])  # (W, D/2)
    H, W = grid.shape[2], grid.shape[3]
    emb_h_tiled = np.repeat(emb_h, W, axis=0)  # (H*W, D/2)
    emb_w_tiled = np.tile(emb_w, (H, 1))       # (H*W, D/2)
    return np.concatenate([emb_h_tiled, emb_w_tiled], axis=1)  # (H*W, D)


# ===================================================================
# DropPath (stochastic depth)
# ===================================================================

class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample.

    During training each sample's residual branch is randomly dropped with
    probability ``drop_prob``.  At test time the output is scaled by the
    keep probability so the expected output is unchanged.
    """

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Work with arbitrary leading dims (B, ..., D)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        output = x / keep_prob * random_tensor
        return output

    def extra_repr(self):
        return f"drop_prob={self.drop_prob:.3f}"


# ===================================================================
# MLP (feed-forward block)
# ===================================================================

class MLP(nn.Module):
    """Two-layer feed-forward network with configurable activation and dropout."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ===================================================================
# Multi-head self-attention
# ===================================================================

class Attention(nn.Module):
    """Multi-head self-attention with fused QKV projection.

    Uses a single ``nn.Linear(dim, dim * 3)`` for Q, K, V projection
    (NOT ``nn.MultiheadAttention``), followed by manual reshape for heads.
    """

    def __init__(self, dim, num_heads=12, qkv_bias=True, attn_drop=0.0,
                 proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, N, D)

        Returns:
            Tuple of (output, attn_weights) where output is (B, N, D) and
            attn_weights is (B, num_heads, N, N).
        """
        B, N, C = x.shape
        qkv = self.qkv(x)  # (B, N, 3*D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv.unbind(0)  # each (B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


# ===================================================================
# Transformer block (pre-norm)
# ===================================================================

class Block(nn.Module):
    """Pre-norm Transformer block: LayerNorm -> Attention -> residual ->
    LayerNorm -> MLP -> residual, with optional DropPath."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True,
                 drop=0.0, attn_drop=0.0, drop_path=0.0,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            drop=drop,
        )

    def forward(self, x):
        y, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ===================================================================
# Patch embedding
# ===================================================================

class PatchEmbed(nn.Module):
    """2-D image to non-overlapping patch embedding using Conv2d."""

    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        if isinstance(img_size, (list, tuple)):
            img_size = (img_size[0], img_size[0]) if len(img_size) == 1 else tuple(img_size[:2])
        elif isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(patch_size, (list, tuple)):
            patch_size = (patch_size[0], patch_size[0]) if len(patch_size) == 1 else tuple(patch_size[:2])
        elif isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, num_patches, embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


# ===================================================================
# Weight initialisation helpers
# ===================================================================

def _init_weights(m):
    """Initialize Linear with truncated normal, LayerNorm with constant 0/1."""
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def _fix_init_weight(model, blocks_attr="blocks"):
    """Rescale attention proj and MLP fc2 by 1 / sqrt(2 * layer_index).

    This stabilises training for deep transformers (following DeepNorm /
    the original I-JEPA recipe).
    """
    blocks = getattr(model, blocks_attr, [])
    for layer_id, block in enumerate(blocks):
        factor = 1.0 / math.sqrt(2.0 * (layer_id + 1))
        block.attn.proj.weight.data.mul_(factor)
        block.mlp.fc2.weight.data.mul_(factor)


# ===================================================================
# Vision Transformer — Encoder
# ===================================================================

class VisionTransformer(nn.Module):
    """I-JEPA context encoder (ViT).

    Produces patch-level representations from a 2-D image.  No CLS token is
    used; positional embeddings are fixed 2-D sinusoidal.

    Args:
        img_size: Input image resolution (int or list).
        patch_size: Patch resolution.
        in_chans: Number of input channels.
        embed_dim: Transformer hidden dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: Hidden-dim multiplier for the MLP.
        qkv_bias: Whether to add bias to QKV projection.
        drop_rate: Dropout rate for attention output and MLP.
        attn_drop_rate: Dropout rate inside attention.
        drop_path_rate: Stochastic depth rate (linearly increased).
        norm_layer: Normalisation layer constructor.
    """

    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # --- Patch embedding ---
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # --- Fixed 2-D sinusoidal positional embedding (no CLS) ---
        pos_embed = get_2d_sincos_pos_embed(embed_dim, self.patch_embed.grid_size)
        self.pos_embed = nn.Parameter(
            torch.from_numpy(pos_embed).float().unsqueeze(0),  # (1, N, D)
            requires_grad=False,
        )

        # --- Transformer blocks with linearly increasing drop-path ---
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # --- Initialise weights ---
        self.apply(_init_weights)
        _fix_init_weight(self, "blocks")

    @property
    def num_patches(self):
        return self.patch_embed.num_patches

    def forward(self, x, masks=None):
        """Forward pass.

        Args:
            x: (B, C, H, W) input image.
            masks: Optional list of index tensors for masking.  Each tensor
                has shape (B, num_keep) and selects a subset of patch tokens
                *after* positional embedding is added.

        Returns:
            (B_total, num_keep, embed_dim) if masks are provided, else
            (B, num_patches, embed_dim).
        """
        # Patch embed
        x = self.patch_embed(x)  # (B, N, D)

        # Add positional embedding
        x = x + self.pos_embed

        # Apply masks (select visible patches) if provided
        if masks is not None:
            x = apply_masks(x, masks)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x


# ===================================================================
# Vision Transformer — Predictor
# ===================================================================

class VisionTransformerPredictor(nn.Module):
    """I-JEPA predictor network.

    Takes the *context* tokens from the encoder and predicts representations
    for the *target* (masked) positions.

    Args:
        num_patches: Total number of patches in the full grid.
        embed_dim: Encoder's embedding dimension.
        predictor_embed_dim: Predictor's internal dimension.
        depth: Number of predictor transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        qkv_bias: QKV bias flag.
        drop_rate: Dropout rate.
        attn_drop_rate: Attention dropout rate.
        drop_path_rate: Stochastic depth rate.
        norm_layer: Normalisation layer constructor.
    """

    def __init__(
        self,
        num_patches,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.predictor_embed_dim = predictor_embed_dim

        # --- Project encoder tokens into predictor dimension ---
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        # --- Learnable mask token ---
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        # --- Fixed sinusoidal positional embedding for all patch positions ---
        # We use 2-D sinusoidal if num_patches is a perfect square, else 1-D.
        grid_side = int(math.sqrt(num_patches))
        if grid_side * grid_side == num_patches:
            pos_embed_np = get_2d_sincos_pos_embed(predictor_embed_dim, grid_side)
        else:
            pos_embed_np = get_1d_sincos_pos_embed(predictor_embed_dim, num_patches)
        self.predictor_pos_embed = nn.Parameter(
            torch.from_numpy(pos_embed_np).float().unsqueeze(0),  # (1, N, Dp)
            requires_grad=False,
        )

        # --- Transformer blocks ---
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        self.predictor_norm = norm_layer(predictor_embed_dim)

        # --- Project back to encoder dimension ---
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        # --- Initialise weights ---
        self.apply(_init_weights)
        _fix_init_weight(self, "predictor_blocks")
        trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x, masks_x, masks):
        """Predict target representations from context tokens.

        Args:
            x: (B, N_ctx, embed_dim) — context tokens from the encoder.
            masks_x: List of index tensors indicating which patch positions
                the context tokens correspond to (used for adding the correct
                positional embeddings).
            masks: List of index tensors indicating the target (masked)
                positions to predict.

        Returns:
            (B_total, N_target, embed_dim) — predicted representations at
            the target positions, projected back to encoder dim.
        """
        assert (masks is not None) and (masks_x is not None), (
            "Predictor requires both masks_x (context positions) and masks "
            "(target positions)."
        )

        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        if not isinstance(masks, list):
            masks = [masks]

        B = x.shape[0] // len(masks_x)  # actual batch size

        # 1. Project context tokens into predictor dimension
        x = self.predictor_embed(x)  # (B_total, N_ctx, Dp)

        # 2. Add positional embeddings to context tokens
        x_pos = self.predictor_pos_embed.repeat(B, 1, 1)  # (B, N_all, Dp)
        x += apply_masks(x_pos, masks_x)  # (B_total, N_ctx, Dp)

        _, N_ctx, Dp = x.shape

        # 3. Create mask tokens for ALL target positions at once
        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        pos_embs = apply_masks(pos_embs, masks)  # (B * npred, N_target, Dp)
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))

        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        pred_tokens += pos_embs

        # 4. Repeat context for each target mask and concatenate
        x = x.repeat(len(masks), 1, 1)  # (B * npred, N_ctx, Dp)
        x = torch.cat([x, pred_tokens], dim=1)  # (B * npred, N_ctx + N_target, Dp)

        # 5. Single transformer pass
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # 6. Extract predictions (mask token outputs only)
        x = x[:, N_ctx:]
        x = self.predictor_proj(x)

        results = [x]

        return torch.cat(results, dim=0)  # (B_total * len(masks), N_target, D)


# ===================================================================
# Slice Encoder — for slice-level I-JEPA
# ===================================================================

class SliceEncoder(nn.Module):
    """Transformer encoder operating on a sequence of slice tokens.

    Unlike :class:`VisionTransformer`, this module does **not** include a
    patch embedding layer.  It expects pre-computed slice features (e.g.
    from a frozen ConvNeXt) and uses 1-D sinusoidal positional embeddings.

    Args:
        num_slices: Number of slice tokens per volume (default 32).
        embed_dim: Token dimension (must match the feature extractor output).
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        qkv_bias: QKV bias flag.
        drop_rate: Dropout rate.
        attn_drop_rate: Attention dropout rate.
        drop_path_rate: Stochastic depth rate.
        norm_layer: Normalisation layer constructor.
    """

    def __init__(
        self,
        num_slices=32,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_slices = num_slices

        # --- Fixed 1-D sinusoidal positional embedding ---
        pos_embed = get_1d_sincos_pos_embed(embed_dim, num_slices)
        self.pos_embed = nn.Parameter(
            torch.from_numpy(pos_embed).float().unsqueeze(0),  # (1, S, D)
            requires_grad=False,
        )

        # --- Transformer blocks ---
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # --- Initialise weights ---
        self.apply(_init_weights)
        _fix_init_weight(self, "blocks")

    @property
    def num_patches(self):
        """Alias so masking code can treat this uniformly with VisionTransformer."""
        return self.num_slices

    def forward(self, x, masks=None):
        """Forward pass.

        Args:
            x: (B, num_slices, embed_dim) — pre-computed slice features.
            masks: Optional list of index tensors (same format as
                VisionTransformer).

        Returns:
            Encoded slice representations.
        """
        # Add positional embedding
        x = x + self.pos_embed

        # Apply masks if provided
        if masks is not None:
            x = apply_masks(x, masks)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x


# ===================================================================
# Slice Predictor — for slice-level I-JEPA
# ===================================================================

class SlicePredictor(nn.Module):
    """Predictor network for slice-level I-JEPA.

    Analogous to :class:`VisionTransformerPredictor` but uses 1-D positional
    embeddings suited for a linear sequence of slices.

    Args:
        num_slices: Number of slice positions.
        embed_dim: Encoder embedding dimension.
        predictor_embed_dim: Predictor internal dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        qkv_bias: QKV bias flag.
        drop_rate: Dropout rate.
        attn_drop_rate: Attention dropout rate.
        drop_path_rate: Stochastic depth rate.
        norm_layer: Normalisation layer constructor.
    """

    def __init__(
        self,
        num_slices=32,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.num_slices = num_slices
        self.predictor_embed_dim = predictor_embed_dim

        # --- Project encoder tokens into predictor dimension ---
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        # --- Learnable mask token ---
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        # --- Fixed 1-D sinusoidal positional embedding ---
        pos_embed_np = get_1d_sincos_pos_embed(predictor_embed_dim, num_slices)
        self.predictor_pos_embed = nn.Parameter(
            torch.from_numpy(pos_embed_np).float().unsqueeze(0),  # (1, S, Dp)
            requires_grad=False,
        )

        # --- Transformer blocks ---
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        self.predictor_norm = norm_layer(predictor_embed_dim)

        # --- Project back to encoder dimension ---
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        # --- Initialise weights ---
        self.apply(_init_weights)
        _fix_init_weight(self, "predictor_blocks")
        trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x, masks_x, masks):
        """Predict target representations from context slice tokens.

        Matches the original I-JEPA VisionTransformerPredictor: all target
        mask tokens are concatenated and processed in a SINGLE transformer
        pass so that mask tokens from different target blocks can attend to
        each other.

        Args:
            x: (B_total, N_ctx, embed_dim) — context tokens from SliceEncoder.
            masks_x: List of index tensors for context slice positions.
            masks: List of index tensors for target (masked) slice positions.

        Returns:
            Predicted representations at target positions, projected to
            encoder dim.  Shape: (B * len(masks), N_target, embed_dim).
        """
        assert (masks is not None) and (masks_x is not None), (
            "SlicePredictor requires both masks_x and masks."
        )

        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        if not isinstance(masks, list):
            masks = [masks]

        B = x.shape[0] // len(masks_x)

        # 1. Project context tokens into predictor dimension
        x = self.predictor_embed(x)  # (B_total, N_ctx, Dp)

        # 2. Add positional embeddings to context tokens
        x_pos = self.predictor_pos_embed.repeat(B, 1, 1)  # (B, S, Dp)
        x += apply_masks(x_pos, masks_x)  # (B_total, N_ctx, Dp)

        _, N_ctx, Dp = x.shape

        # 3. Create mask tokens for ALL target positions at once
        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        pos_embs = apply_masks(pos_embs, masks)  # (B * npred, N_target, Dp)
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))

        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        pred_tokens += pos_embs

        # 4. Repeat context for each target mask and concatenate
        x = x.repeat(len(masks), 1, 1)  # (B * npred, N_ctx, Dp)
        x = torch.cat([x, pred_tokens], dim=1)  # (B * npred, N_ctx + N_target, Dp)

        # 5. Single transformer pass
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # 6. Extract predictions (mask token outputs only)
        x = x[:, N_ctx:]
        x = self.predictor_proj(x)

        return x


# ===================================================================
# Factory functions
# ===================================================================

def vit_base(patch_size=16, **kwargs):
    """Create a ViT-Base encoder for I-JEPA.

    Default: embed_dim=768, depth=12, num_heads=12.
    """
    defaults = dict(
        img_size=256,
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
    )
    defaults.update(kwargs)
    return VisionTransformer(**defaults)


def vit_predictor(**kwargs):
    """Create a ViT predictor for I-JEPA.

    Default: predictor_embed_dim=384, depth=6, num_heads=12.
    """
    defaults = dict(
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
    )
    defaults.update(kwargs)
    return VisionTransformerPredictor(**defaults)


def slice_encoder(**kwargs):
    """Create a SliceEncoder for slice-level I-JEPA.

    Default: num_slices=32, embed_dim=768, depth=12, num_heads=12.
    """
    defaults = dict(
        num_slices=32,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
    )
    defaults.update(kwargs)
    return SliceEncoder(**defaults)


def slice_predictor(**kwargs):
    """Create a SlicePredictor for slice-level I-JEPA.

    Default: num_slices=32, embed_dim=768, predictor_embed_dim=384,
    depth=6, num_heads=12.
    """
    defaults = dict(
        num_slices=32,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
    )
    defaults.update(kwargs)
    return SlicePredictor(**defaults)
