import numpy as np
import pandas as pd

from ..util.error_message import INTERFACE_ERROR_MSG

from .object_detector import ObjectDetector


class BlobDetector(ObjectDetector):
    def detect(self, image: np.ndarray) -> pd.DataFrame:
        raise NotImplementedError(INTERFACE_ERROR_MSG)
