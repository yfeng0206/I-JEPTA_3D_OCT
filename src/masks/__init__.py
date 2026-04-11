from src.masks.utils import apply_masks
from src.masks.multiblock import MaskCollator

try:
    from src.masks.slice_mask import SliceMaskCollator
except ImportError:
    SliceMaskCollator = None  # Slice-level approach (archived)
