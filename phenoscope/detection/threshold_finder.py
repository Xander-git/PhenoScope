import numpy as np
import pandas as pd

from ._threshold_finder_view import ThresholdFinderView

"""
End-point class for ObjectFinder
"""


class ThresholdFinder(ThresholdFinderView):
    def find_objects(self, image: np.ndarray) -> pd.DataFrame:
        super().find_objects(image=image)
        return self.results
