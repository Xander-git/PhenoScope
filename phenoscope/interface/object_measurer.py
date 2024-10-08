import pandas as pd
import numpy as np

from ._image_operation import ImageOperation
from ..util.error_message import INTERFACE_ERROR_MSG


# <<Interface>>
class ObjectMeasurer(ImageOperation):
    def __init__(self):
        raise NotImplementedError(INTERFACE_ERROR_MSG)

    def measure(self, image:np.ndarray,object_map: np.ndarray, object_table: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(INTERFACE_ERROR_MSG)
