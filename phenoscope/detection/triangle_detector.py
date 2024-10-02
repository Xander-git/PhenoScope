import numpy as np

from skimage.filters import threshold_triangle

from ..interface import ThresholdDetector


class TriangleDetector(ThresholdDetector):
    def __init__(self):
        pass

    def detect(self, image: np.ndarray) -> np.ndarray:
        return image >= threshold_triangle(image)
