from phenoscope.pipeline.interface import PreprocessorPipeline
from phenoscope.preprocessing import CLAHE, MedianFilter
from phenoscope.morphology import WhiteTophatMorpher

class ClaheBoostPreprocessor(PreprocessorPipeline):
    def __init__(self, kernel_size=None, tophat_shape='disk', tophat_radius=None):
        super().__init__(pipeline_queue={
            'CLAHE':CLAHE(kernel_size=kernel_size),
            'White Tophat': WhiteTophatMorpher(
                    footprint_shape=tophat_shape,
                    footprint_radius=tophat_radius
            ),
            'Median Filter': MedianFilter()
        })