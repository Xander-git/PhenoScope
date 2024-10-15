from ._image_operation import ImageOperation
from ..util.error_message import INTERFACE_ERROR_MSG
from .. import Image


class ImageTransformer(ImageOperation):

    def transform(self, image: Image) -> Image:
        output = self._operate(image)

    def _operate(self, input: Image) -> Image:
        raise NotImplementedError(INTERFACE_ERROR_MSG)
