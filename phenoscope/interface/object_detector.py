from ._image_operation import ImageOperation
from ..util.error_message import INTERFACE_ERROR_MSG, ARRAY_CHANGE_ERROR_MSG, ENHANCED_ARRAY_CHANGE_ERROR_MSG
from .. import Image


# <<Interface>>
class ObjectDetector(ImageOperation):
    def detect(self, image: Image) -> Image:
        input = image.copy()
        output = self._operate(image)

        # Post Operation Checks
        if input.array != output.array: raise AttributeError(ARRAY_CHANGE_ERROR_MSG)
        if input.enhanced_array != output.enhanced_array: raise AttributeError(ENHANCED_ARRAY_CHANGE_ERROR_MSG)

        return output

    def _operate(self, image: Image) -> Image:
        raise NotImplementedError(INTERFACE_ERROR_MSG)
