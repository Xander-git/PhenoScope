import logging

formatter = logging.Formatter(
        fmt=f'[%(asctime)s|%(name)s] %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S'
)
console_handler = logging.StreamHandler()
log = logging.getLogger(__name__)
log.addHandler(console_handler)
console_handler.setFormatter(formatter)

import numpy as np

from .imaging_pipeline import ImagingPipeline


class PreprocessorPipeline(ImagingPipeline):
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if type(image) is not np.ndarray:
            raise ValueError("Input of a ImagePreprocessorPipeline must be a digitized image array")

        output = self._execute_pipeline(input=image)

        if type(output) == np.ndarray:
            return output
        else:
            raise ValueError("Final output of a ImagePreprocessorPipeline must be an image array")
