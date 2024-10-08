import numpy as np
from scipy.ndimage import binary_fill_holes

from ..interface import MorphologyMorpher
from ..util.type_checks import is_binary_mask


class MaskFillMorpher(MorphologyMorpher):
    def __init__(self, structure: np.ndarray = None, origin: int = 0):
        if not (is_binary_mask(structure)): raise ValueError('input object array must be a binary array')
        self._structure = structure
        self._origin = origin

    def morph(self, object_mask: np.ndarray) -> np.ndarray:
        if not (is_binary_mask(object_mask)): raise ValueError('input object array must be a binary array')
        return binary_fill_holes(
                input=object_mask,
                structure=self._structure,
                origin=self._origin
        )
