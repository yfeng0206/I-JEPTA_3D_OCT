"""Frozen feature extractor for slice-level I-JEPA.

Loads a ConvNeXt backbone (optionally from a SLIViT checkpoint) and
extracts a 768-dimensional feature vector per OCT slice.  All parameters
are frozen after loading.

Compatible with PyTorch 1.13.1 and the ``transformers`` library.
"""

import logging
from collections import OrderedDict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FrozenFeatureExtractor(nn.Module):
    """Frozen ConvNeXt backbone that produces a 768-d vector per slice.

    The model is loaded from HuggingFace (``ConvNextModel``) and optionally
    fine-tuned weights from a SLIViT / Kermany checkpoint are overlaid.
    Only the embedding and encoder stages are kept; the classification head
    (if any) is discarded.

    After construction **all** parameters are frozen and the module is set
    to eval mode.

    Args:
        checkpoint_path: Optional path to a SLIViT-style ``*.pt`` checkpoint
            that contains ConvNeXt weights under keys prefixed with
            ``"model.convnext."``.  If ``None`` the default HuggingFace
            pretrained weights are used.
        convnext_name: HuggingFace model identifier for ConvNeXt.  The
            default (``"facebook/convnext-tiny-224"``) produces 768-d
            features after global average pooling.
    """

    def __init__(self, checkpoint_path=None,
                 convnext_name="facebook/convnext-tiny-224",
                 freeze=True):
        super().__init__()
        self._freeze = freeze

        # ------------------------------------------------------------------
        # 1. Load ConvNextModel from HuggingFace
        # ------------------------------------------------------------------
        try:
            from transformers import ConvNextModel
        except ImportError:
            raise ImportError(
                "The `transformers` library is required for FrozenFeatureExtractor. "
                "Install it with:  pip install transformers"
            )

        convnext_full = ConvNextModel.from_pretrained(convnext_name)

        # Extract the first two children: embeddings + encoder
        # ConvNextModel has: embeddings, encoder, layernorm (we keep first two)
        children = list(convnext_full.children())
        # children[0] = ConvNextEmbeddings  (stem / patch embed)
        # children[1] = ConvNextEncoder     (stages)
        self.backbone = nn.Sequential(
            children[0],  # embeddings
            children[1],  # encoder
        )

        # The layernorm from ConvNextModel (applied to last hidden state)
        # We keep it for proper feature normalization
        if len(children) > 2:
            self.final_layernorm = children[2]
        else:
            self.final_layernorm = nn.Identity()

        # ------------------------------------------------------------------
        # 2. Optionally load SLIViT / Kermany checkpoint weights
        # ------------------------------------------------------------------
        if checkpoint_path is not None:
            self._load_slivit_checkpoint(checkpoint_path)

        # ------------------------------------------------------------------
        # 3. Freeze parameters if requested
        # ------------------------------------------------------------------
        if freeze:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False

        logger.info(
            "FeatureExtractor initialised from '%s'%s — %s.",
            convnext_name,
            f" + checkpoint '{checkpoint_path}'" if checkpoint_path else "",
            "frozen" if freeze else "trainable (low LR recommended)",
        )

    def _load_slivit_checkpoint(self, checkpoint_path):
        """Load and remap SLIViT checkpoint weights into the backbone.

        SLIViT checkpoints store ConvNeXt weights under keys like:
            ``model.convnext.embeddings.…``  and  ``model.convnext.encoder.…``

        These need to be remapped to the ``nn.Sequential`` indices used by
        ``self.backbone``:
            ``model.convnext.embeddings.…`` -> ``0.…``
            ``model.convnext.encoder.…``    -> ``1.…``
        """
        raw_state = torch.load(checkpoint_path, map_location="cpu")

        # Some checkpoints wrap the state dict under a key
        if "state_dict" in raw_state:
            raw_state = raw_state["state_dict"]
        elif "model_state_dict" in raw_state:
            raw_state = raw_state["model_state_dict"]
        elif "model" in raw_state and isinstance(raw_state["model"], dict):
            raw_state = raw_state["model"]

        remapped = OrderedDict()
        prefix = "model.convnext."

        for key, value in raw_state.items():
            if not key.startswith(prefix):
                continue

            suffix = key[len(prefix):]

            # Remap embeddings -> 0, encoder -> 1
            if suffix.startswith("embeddings."):
                new_key = "0." + suffix[len("embeddings."):]
            elif suffix.startswith("encoder."):
                new_key = "1." + suffix[len("encoder."):]
            else:
                # Skip keys that don't belong to embeddings/encoder
                # (e.g. layernorm, pooler, classifier)
                logger.debug("Skipping SLIViT key: %s", key)
                continue

            remapped[new_key] = value

        if not remapped:
            logger.warning(
                "No keys matched prefix '%s' in checkpoint '%s'. "
                "Available keys (first 10): %s",
                prefix, checkpoint_path,
                list(raw_state.keys())[:10],
            )
            return

        missing, unexpected = self.backbone.load_state_dict(remapped, strict=False)
        if missing:
            logger.warning("Missing keys when loading SLIViT weights: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys when loading SLIViT weights: %s", unexpected)
        logger.info(
            "Loaded %d parameters from SLIViT checkpoint '%s'.",
            len(remapped), checkpoint_path,
        )

    def train(self, mode=True):
        """If frozen, keep permanently in eval mode."""
        if self._freeze:
            return super().train(False)
        return super().train(mode)

    def forward(self, x):
        """Extract a 768-d feature vector from a single 2-D slice.

        Args:
            x: (B, 3, 256, 256) — single OCT slice (RGB or 3-channel).

        Returns:
            (B, 768) — global-average-pooled feature vector.
        """
        # ConvNextEmbeddings expects pixel_values but when wrapped in
        # nn.Sequential we call it directly.  The embeddings module's
        # forward takes (pixel_values) and returns (B, C, H, W).
        features = self.backbone[0](x)  # embeddings -> (B, C_init, H', W')

        # The encoder expects hidden_states (B, C, H, W) and returns
        # a BaseModelOutputWithNoAttention.  We need the last_hidden_state.
        encoder_out = self.backbone[1](features)

        # Handle both raw tensor and HuggingFace output objects
        if hasattr(encoder_out, "last_hidden_state"):
            hidden = encoder_out.last_hidden_state  # (B, C, H, W)
        elif isinstance(encoder_out, tuple):
            hidden = encoder_out[0]
        else:
            hidden = encoder_out

        # Apply final layer norm if present
        if not isinstance(self.final_layernorm, nn.Identity):
            hidden = self.final_layernorm(hidden)

        # Global average pool: (B, 768, H, W) -> (B, 768)
        if hidden.dim() == 4:
            pooled = hidden.mean(dim=[2, 3])
        elif hidden.dim() == 3:
            # Some versions return (B, N, C) — pool over sequence dim
            pooled = hidden.mean(dim=1)
        else:
            pooled = hidden

        return pooled

    @torch.no_grad()
    def encode_volume(self, slices):
        """Encode every slice of an OCT volume independently.

        Args:
            slices: (B, num_slices, 3, 256, 256) — a batch of OCT volumes,
                each containing ``num_slices`` 2-D slices.

        Returns:
            (B, num_slices, 768) — per-slice feature vectors.
        """
        B, S, C, H, W = slices.shape

        # Flatten to (B*S, C, H, W), encode, reshape back
        flat = slices.reshape(B * S, C, H, W)
        features = self.forward(flat)  # (B*S, 768)
        features = features.reshape(B, S, -1)  # (B, S, 768)

        return features
