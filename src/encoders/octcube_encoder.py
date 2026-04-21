"""OCTCube: 3D OCT foundation model (Liu 2024, Nature Comms 2025).

Weights: HuggingFace zucksliu/OCTCubeM  + Google Drive (mirror).
Code:    https://github.com/ZucksLiu/OCTCubeM  (expected to be git-cloned
         into a local directory and added to sys.path at runtime; see the
         AML shell script).

Architecture details from their models_vit_st_flash_attn_slivit.py:
  img_size=256, patch_size=16, in_chans=1 (gray), embed_dim=768, depth=12,
  num_heads=12, num_frames=60, t_patch_size variable.

Input layout required by OCTCube:  (B, 1, num_frames, H, W) grayscale.
Our OCTVolumeDataset gives us (S, 3, H, W) with S slices in [0, 1].

Adapter conversions per volume:
  (S=64, 3, 256, 256)  →  (1, num_frames=60, 256, 256) gray
                      →  (1, 1, 60, 256, 256) matched to OCTCube input

If S != num_frames, we either center-crop or linearly interpolate along the
temporal dim; we default to 'interp' since our 64-slice subset is already a
uniform linspace through the 200-slice native volume.

Output: volume-level embedding (D,) after the encoder's global-pool or CLS
token (whichever the checkpoint uses).

Notes / caveats:
  - Flash-attn may or may not be available in the runtime env; OCTCube has
    a non-flash-attn `models_vit.py` we can fall back to. We detect at load
    time.
  - The OCTCube pretrained weights come in several variants; we default to
    their plain `OCTCube.pth` (pre-trained, not fine-tuned).
  - This adapter requires `OCTCUBE_REPO_DIR` env var pointing at the cloned
    repo (handled by the shell driver).
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from . import register
from .base import EncoderAdapter


@register
class OCTCubeEncoder(EncoderAdapter):
    name = 'octcube'
    embed_dim = 768
    input_layout = 'volume_3d'
    input_size = 256
    # gray-scale normalization — OCTCube uses simple [0, 1] scaling; no
    # ImageNet mean/std. Override the parent's ImageNet defaults.
    imagenet_mean = (0.0, 0.0, 0.0)
    imagenet_std  = (1.0, 1.0, 1.0)

    DEFAULT_NUM_FRAMES = 60
    DEFAULT_T_PATCH_SIZE = 3

    def _load_model(self):
        repo_dir = os.environ.get('OCTCUBE_REPO_DIR',
                                  self.kwargs.get('octcube_repo_dir'))
        if repo_dir is None or not os.path.isdir(repo_dir):
            raise RuntimeError(
                'OCTCube adapter needs OCTCUBE_REPO_DIR env var (or '
                "octcube_repo_dir kwarg) pointing at a clone of "
                'https://github.com/ZucksLiu/OCTCubeM. Run `git clone ...` '
                'first and set the path.'
            )
        # Their repo has models under OCTCubeM/OCTCube/.
        octcube_src = os.path.join(repo_dir, 'OCTCube')
        if octcube_src not in sys.path:
            sys.path.insert(0, octcube_src)

        # Pick the right VisionTransformer module:
        #   - models_vit_st_flash_attn.py hard-imports `flash_attn` at module
        #     level (line 47), so just importing it fails without the package.
        #     Only use it if flash_attn is actually installed.
        #   - models_vit_st.py is the non-FA spatiotemporal variant. It does
        #     NOT accept a `use_flash_attn` kwarg.
        # Both variants have the same __init__ param list otherwise and
        # produce a compatible embedding when we stub out the classifier
        # head (see below).
        use_flash = self.kwargs.get('use_flash_attn', False)
        VisionTransformer = None
        model_kind = None
        if use_flash:
            try:
                import flash_attn  # noqa: F401
                from models_vit_st_flash_attn import VisionTransformer as _VT
                VisionTransformer, model_kind = _VT, 'st_flash_attn'
            except ImportError:
                print('[octcube] flash_attn unavailable at import time -> '
                      'falling back to models_vit_st (non-FA)')
                use_flash = False
        if VisionTransformer is None:
            from models_vit_st import VisionTransformer as _VT
            VisionTransformer, model_kind = _VT, 'st'

        num_frames = int(self.kwargs.get('num_frames', self.DEFAULT_NUM_FRAMES))
        t_patch_size = int(self.kwargs.get('t_patch_size', self.DEFAULT_T_PATCH_SIZE))

        self.num_frames = num_frames
        self.t_patch_size = t_patch_size
        self.model_kind = model_kind

        init_kwargs = dict(
            num_frames=num_frames,
            t_patch_size=t_patch_size,
            img_size=self.input_size,
            patch_size=16,
            in_chans=1,
            num_classes=1,                   # dummy 1-class head (we stub it to Identity below)
            embed_dim=self.embed_dim,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            sep_pos_embed=True,
            cls_embed=True,
            global_pool=True,                # forward mean-pools patch tokens
        )
        # use_flash_attn is only accepted by the FA variant.
        if model_kind == 'st_flash_attn':
            init_kwargs['use_flash_attn'] = use_flash

        self.model = VisionTransformer(**init_kwargs).to(self.device).eval()

        # Override head + dropout so forward(x) returns the raw embedding.
        # Both variants end their forward with `self.head(self.dropout(x))` —
        # setting both to Identity makes them return the pre-head (B, D)
        # embedding directly, regardless of the FA vs non-FA return signature.
        self.model.head = nn.Identity()
        if hasattr(self.model, 'dropout'):
            self.model.dropout = nn.Identity()

        if self.weights_path is None:
            raise RuntimeError('OCTCube weights_path must be provided (path to OCTCube.pth)')
        ckpt = torch.load(self.weights_path, map_location='cpu', weights_only=False)
        # OCTCube weights usually come as {'model': state_dict} or as a
        # plain state_dict. Handle both.
        sd = ckpt.get('model', ckpt)
        # Strip common prefixes.
        sd = {k.replace('module.', '').replace('encoder.', ''): v for k, v in sd.items()}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        print(f'[octcube] loaded weights: {len(missing)} missing, '
              f'{len(unexpected)} unexpected keys')

    @torch.no_grad()
    def encode_volume(self, volume):
        """volume: (S, 3, H, W) in [0, 1]. Returns (D,) float16 CPU."""
        # 1. RGB → grayscale (average channels since all three are the same
        #    replicated gray channel in OCT preprocessing).
        gray = volume.mean(dim=1, keepdim=True)              # (S, 1, H, W)
        # 2. Resize if needed (already 256 in our pipeline).
        gray = self._resize_if_needed(gray)                  # (S, 1, 256, 256)
        # 3. Interpolate temporal dim from S → num_frames if mismatched.
        if gray.size(0) != self.num_frames:
            gray = gray.permute(1, 0, 2, 3).unsqueeze(0)     # (1, 1, S, H, W)
            gray = F.interpolate(
                gray, size=(self.num_frames, self.input_size, self.input_size),
                mode='trilinear', align_corners=False,
            )                                                 # (1, 1, F, H, W)
        else:
            gray = gray.permute(1, 0, 2, 3).unsqueeze(0)     # (1, 1, F, H, W)
        gray = gray.to(self.device)

        with autocast():
            # head + dropout have been stubbed to Identity in _load_model,
            # so forward(x) returns the (B, D) mean-pooled embedding for
            # both FA and non-FA variants. No return_embeddings flag
            # needed (non-FA variant doesn't have one anyway).
            embedding = self.model(gray)
        feats = embedding.squeeze(0) if embedding.dim() == 2 else embedding
        return feats.to(torch.float16).cpu()
