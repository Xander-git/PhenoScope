import numpy as np

from ._image_operation import ImageOperation
from ..util.error_message import INTERFACE_ERROR_MSG


class ImagePreprocessor(ImageOperation):
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError(INTERFACE_ERROR_MSG)
