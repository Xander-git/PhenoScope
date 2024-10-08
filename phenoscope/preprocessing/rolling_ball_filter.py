import numpy as np
from skimage.restoration import rolling_ball

from ..interface import ImagePreprocessor


class RollingBallFilter(ImagePreprocessor):
    def __init__(self, radius: int = 100, kernel: np.ndarray = None, nansafe: bool = False, num_threads: int = None):
        self._radius: int = radius
        self._kernel: np.ndarray = kernel
        self._nansafe: bool = nansafe
        self._num_threads: int = num_threads

    def preprocess(self, image: np.ndarray):
        return image - rolling_ball(image=image,
                                    radius=self._radius,
                                    kernel=self._kernel,
                                    nansafe=self._nansafe,
                                    num_threads=self._num_threads)
