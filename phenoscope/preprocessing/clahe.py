import numpy as np
from skimage.exposure import equalize_adapthist

from ..interface import ContrastPreprocessor


class CLAHE(ContrastPreprocessor):
    def __init__(self, kernel_size: int = None):
        self.__kernel_size: int = kernel_size

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if self.__kernel_size is None:
            return equalize_adapthist(
                    image=image,
                    kernel_size=int(min(image.shape[:1]) * (1.0 / 15.0))
            )
        else:
            return equalize_adapthist(
                    image=image,
                    kernel_size=self.__kernel_size
            )
