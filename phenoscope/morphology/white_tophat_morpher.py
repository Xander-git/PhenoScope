import numpy as np
from skimage.morphology import disk, square, white_tophat, cube, ball

from ..interface import MorphologyMorpher
from ..util.type_checks import is_binary_mask


class WhiteTophatMorpher(MorphologyMorpher):
    def __init__(self, footprint_shape='disk', footprint_radius: int = None):
        self._footprint_shape = footprint_shape
        self._footprint_radius = footprint_radius

    def morph(self, image: np.ndarray) -> np.ndarray:
        white_tophat_results = white_tophat(image, footprint=self._get_footprint(self._get_footprint_radius(image=image)))
        if is_binary_mask(image):
            return image & ~white_tophat_results
        else:
            return image - white_tophat_results

    def _get_footprint_radius(self, image: np.ndarray) -> int:
        if self._footprint_radius is None:
            return int(np.min(image.shape) * 0.002)
        else:
            return self._footprint_radius

    def _get_footprint(self, radius: int) -> np.ndarray:
        match self._footprint_shape:
            case 'disk':
                return disk(radius=radius)
            case 'square':
                return square(radius * 2)
            case 'sphere':
                return ball(radius)
            case 'cube':
                return cube(radius * 2)
