import numpy as np

from ._image_operation import ImageOperation
from ..util.error_message import INTERFACE_ERROR_MSG, PREPROCESSOR_ARRAY_CHANGE_ERROR
from .. import Image


class ImagePreprocessor(ImageOperation):
    def __init__(self):
        pass

    def preprocess(self, image: Image) -> Image:
        input_image: Image = image.copy()
        output = self._operate(input)
        if input_image.array != output.array: raise AttributeError(PREPROCESSOR_ARRAY_CHANGE_ERROR)

        return output

    def _operate(self, image: Image) -> Image:
        raise NotImplementedError(INTERFACE_ERROR_MSG)
