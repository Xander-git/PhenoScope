from ..util.error_message import INTERFACE_ERROR_MSG

from .. import Image

class ImageOperation:
    def __init__(self):
        raise NotImplementedError(INTERFACE_ERROR_MSG)

    def _operate(self, image:Image)->Image:
        raise NotImplementedError(INTERFACE_ERROR_MSG)