import logging

formatter = logging.Formatter(
        fmt=f'[%(asctime)s|%(name)s] %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S'
)
console_handler = logging.StreamHandler()
log = logging.getLogger(__name__)
log.addHandler(console_handler)
console_handler.setFormatter(formatter)

from typing import Dict, Any

from ...interface._image_operation import ImageOperation

from ...interface.object_detector import ObjectDetector
from ...interface.image_preprocessor import ImagePreprocessor
from ...interface.morphology_morpher import MorphologyMorpher
from ...interface.image_transformer import ImageTransformer
from ...interface.measurement_filter import MeasurementFilter
from ...interface.object_measurer import ObjectMeasurer

from ...util.error_message import INTERFACE_ERROR_MSG


class ImagingPipeline(ImageOperation):
    def __init__(self, pipeline_queue: Dict[str, ImageOperation]):
        self._operational_queue: Dict[str, ImageOperation] = pipeline_queue

    def _execute_pipeline(self, input: Any) -> Any:
        output = input
        for key in self._operational_queue.keys():
            log.info(f'Executing image operation named: {key}\n')
            output = self._execute_operation(output, self._operational_queue[key])

        return output

    def _execute_operation(self, input: Any, operator: ImageOperation) -> Any:
        match operator:
            case ImagePreprocessor():
                return operator.preprocess(input)
            case MorphologyMorpher():
                return operator.morph(input)
            case ObjectDetector():
                return operator.detect(input)
            case ObjectMeasurer():
                return operator.measure(input)
            case MeasurementFilter():
                return operator.filter(input)
            case ObjectMeasurer():
                return operator.measure(input)
            case _:
                raise TypeError(f'Operator {type(operator)} not recognized')
