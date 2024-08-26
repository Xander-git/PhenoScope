import sys
import logging

formatter = logging.Formatter(
        fmt=f'[%(asctime)s|%(name)s] %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S'
)
console_handler = logging.StreamHandler()
log = logging.getLogger(__name__)
log.addHandler(console_handler)
console_handler.setFormatter(formatter)

import pandas as pd
import numpy as np

from ._colony_profile_plot_object import ColonyProfilePlotObject


class ColonyProfileMeasureBase(ColonyProfilePlotObject):
    def __init__(self, img: np.ndarray, sample_name: str,
                 auto_run: bool = True,
                 use_boosted_mask: bool = True,
                 boost_kernel_size: bool = None,
                 boost_footprint_radius: bool = 5
                 ):
        self._measurements = {}
        super().__init__(
                img=img,
                sample_name=sample_name,
                auto_run=auto_run,
                use_boosted_mask=use_boosted_mask,
                boost_kernel_size=boost_kernel_size,
                boost_footprint_radius=boost_footprint_radius
        )

    def run(self,
            threshold_method="otsu", use_boosted=True,
            filter_property="distance_from_center", filter_type="min"
            ):
        self.find_colony(threshold_method=threshold_method, use_boosted=use_boosted,
                         filter_property=filter_property, filter_type=filter_type)
        self.measure_colony()

    @property
    def _results(self):
        return self.measurements.copy()

    @property
    def measurements(self):
        return pd.Series(data=self._measurements.values(),
                         index=self._measurements.keys(),
                         name=self.colony_name)

    def measure_colony(self):
        try:
            self._measure_colony()
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except:
            log.warning(f"could not measure colony for sample: {self.sample_name}", exc_info=True)
            self.status_validity = False

    def _measure_colony(self):
        """
        Except when changing the exception handling protocol of the module,
        change this class when adding new measurements to maintain intended exception handling protocol
        :return:
        """
        self._measure_img_dims()

    def _measure_img_dims(self):
        if len(self.input_img.shape) == 3:
            img_height, img_width, _ = self.input_img.shape
        elif len(self.input_img.shape) == 2:
            img_height, img_width = self.input_img.shape
        else:
            raise ValueError("Invalid image input")
        self._measurements["ImgHeight"] = img_height
        self._measurements["ImgWidth"] = img_width
