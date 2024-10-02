import numpy as np
from .object_detector import ObjectDetector
from ..util.error_message import INTERFACE_ERROR_MSG


# <<Interface>>
class ThresholdDetector(ObjectDetector):
    def detect(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError(INTERFACE_ERROR_MSG)
