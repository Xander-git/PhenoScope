import numpy as np

from skimage.filters import threshold_otsu

from ..interface import ThresholdDetector


class OtsuDetector(ThresholdDetector):
    def __init__(self):
        pass

    def detect(self, image: np.ndarray) -> np.ndarray:
        return image >= threshold_otsu(image)
