import numpy as np

from ._image_operation import ImageOperation

from .. import Image
from ..util.error_message import INTERFACE_ERROR_MSG, ARRAY_CHANGE_ERROR_MSG, ENHANCED_ARRAY_CHANGE_ERROR_MSG


# <<Interface>>
class ObjectLinker(ImageOperation):
    def link(self, image: Image) -> Image:
        input_image = image.copy()

        output = self._operate(image)

        if input_image.array != output.array: raise AttributeError(ARRAY_CHANGE_ERROR_MSG)
        if input_image.enhanced_array != output.enhanced_array: raise AttributeError(ENHANCED_ARRAY_CHANGE_ERROR_MSG)

        return output

    def _operate(self, image: Image) -> Image:
        raise NotImplementedError(INTERFACE_ERROR_MSG)
