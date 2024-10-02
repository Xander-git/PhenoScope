import numpy as np

from ..interface import NoisePreprocessor

from skimage.filters import median


class MedianFilter(NoisePreprocessor):
    def __init__(self, mode='nearest', cval: float = 0.0):
        if mode in ['nearest', 'reflect', 'constant', 'mirror', 'wrap']:
            self._mode = mode
            self._cval = cval
        else:
            raise ValueError('mode must be one of "nearest","reflect","constant","mirror","wrap"')

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return median(image=image, behavior='ndimage', mode=self._mode, cval=self._cval)
