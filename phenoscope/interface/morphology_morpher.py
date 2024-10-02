from ._image_operation import ImageOperation

from ..util.error_message import INTERFACE_ERROR_MSG


# <<Interface>>
class MorphologyMorpher(ImageOperation):
    def __init__(self):
        raise NotImplementedError(INTERFACE_ERROR_MSG)

    def transform(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError(INTERFACE_ERROR_MSG)
