import logging

formatter = logging.Formatter(
        fmt=f'[%(asctime)s|%(name)s] %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S'
)
console_handler = logging.StreamHandler()
log = logging.getLogger(__name__)
log.addHandler(console_handler)
console_handler.setFormatter(formatter)


from .imaging_pipeline import ImagingPipeline
from ... import Image


class PreprocessorPipeline(ImagingPipeline):
    def preprocess(self, image: Image) -> Image:
        output = self._execute_pipeline(input=image)
        return output
