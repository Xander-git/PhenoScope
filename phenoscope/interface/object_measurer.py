import pandas as pd

from ._image_operation import ImageOperation
from ..util.error_message import INTERFACE_ERROR_MSG


# <<Interface>>
class ObjectMeasurer(ImageOperation):
    def __init__(self):
        raise NotImplementedError(INTERFACE_ERROR_MSG)

    def measure(self, object_table: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(INTERFACE_ERROR_MSG)
