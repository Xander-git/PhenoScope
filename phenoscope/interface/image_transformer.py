from ._image_operation import ImageOperation
from ..util.error_message import INTERFACE_ERROR_MSG


class ImageTransformer(ImageOperation):
    def __init__(self):
        raise NotImplementedError(INTERFACE_ERROR_MSG)
