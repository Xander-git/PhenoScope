import pandas as pd

from .object_detector import ObjectDetector
from .image_preprocessor import ImagePreprocessor
from .feature_extractor import FeatureExtractor
from .morphology_morpher import MorphologyMorpher
from .object_filter import ObjectFilter
from .object_modifier import ObjectModifier

from .. import Image
from ..util.error_message import INTERFACE_ERROR_MSG


class ObjectProfiler:
    def __init__(
            self,
            detector: ObjectDetector,
            preprocessor: ImagePreprocessor = None,
            morpher: MorphologyMorpher = None,
            measurer: FeatureExtractor = None,
            linker: ObjectModifier = None,
            measurement_filter: ObjectFilter = None
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
        self._measurer: FeatureExtractor = measurer
        self._object_linker: ObjectModifier = linker
        self._measurement_filter: ObjectFilter = measurement_filter

    def profile(self, image: Image) -> pd.DataFrame:
        raise NotImplementedError(INTERFACE_ERROR_MSG)
