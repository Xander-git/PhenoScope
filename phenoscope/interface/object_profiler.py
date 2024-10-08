import pandas as pd
import numpy as np
from dash.html import Table

from .object_detector import ObjectDetector
from .image_preprocessor import ImagePreprocessor
from .object_measurer import ObjectMeasurer
from .morphology_morpher import MorphologyMorpher
from .measurement_filter import MeasurementFilter

from ..util.error_message import INTERFACE_ERROR_MSG


class ObjectProfiler:
    def __init__(
            self,
            detector: ObjectDetector,
            preprocessor: ImagePreprocessor = None,
            morpher: MorphologyMorpher = None,
            measurer: ObjectMeasurer = None,
            measurement_filter: MeasurementFilter = None
    ):
        self._object_table = pd.DataFrame(
                data={
                    'Location_CenterRR': [],
                    'Location_CenterCC': [],
                    'Boundary_Radius'  : []
                }
        )

        self._detector: ObjectDetector = detector
        self._preprocessor: ImagePreprocessor = preprocessor
        self._morpher: MorphologyMorpher = morpher
        self._measurer: ObjectMeasurer = measurer
        self._measurement_filter: MeasurementFilter = measurement_filter

    def profile(self, image: np.ndarray) -> pd.DataFrame:
        raise NotImplementedError(INTERFACE_ERROR_MSG)
