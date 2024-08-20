import logging
import sys

formatter = logging.Formatter(
        fmt=f'[%(asctime)s|%(name)s] %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S'
)
console_handler = logging.StreamHandler()
log = logging.getLogger(__name__)
log.addHandler(console_handler)
console_handler.setFormatter(formatter)

from copy import deepcopy
import pandas as pd

from ..cellprofiler_api import CellProfilerApi
from ._colony_profile_measure import ColonyProfileMeasure

##############################################################################
cp_connection = CellProfilerApi()


class ColonyProfileCellProfilerIntegration(ColonyProfileMeasure):
    """
    Due to how the API works, this class should be the 2nd to last endpoint for the ColonyProfile Class
    """

    def __init__(self, img, sample_name, auto_run=True):
        self._cp_results = None
        super().__init__(img=img, sample_name=sample_name, auto_run=auto_run)

    @property
    def _results(self):
        return pd.concat([self._cp_results, self.measurements], axis=0)

    def run_analysis(self,
                     threshold_method="otsu", use_boosted=True,
                     filter_property="distance_from_center", filter_type="min"
                     ):
        self.find_colony(threshold_method=threshold_method, use_boosted=use_boosted,
                         filter_property=filter_property, filter_type=filter_type)
        self.measure_colony()
        self.run_cp_analysis()

    def run_cp_analysis(self):
        if self.status_validity:
            # TODO: FINISH THIS FUNCTION
            object_measurements = []
            bg_measurements = []
            cp_connection.add_img(self.gray_img, self.sample_name)

            cp_connection.add_object(self.colony_mask, self.colony_name, self.sample_name)
            object_measurements.append(cp_connection.measure_areashape(self.colony_name))
            object_measurements.append(cp_connection.measure_intensity(
                    object_name=self.colony_name,
                    image_name=self.sample_name
            ))
            object_measurements.append(cp_connection.measure_texture(
                    object_name=self.colony_name,
                    image_name=self.sample_name
            ))

            cp_connection.add_object(self.background_mask, self.background_name, self.sample_name)
            bg_measurements.append(cp_connection.measure_areashape(self.background_name))
            bg_measurements.append(cp_connection.measure_intensity(
                    object_name=self.background_name,
                    image_name=self.sample_name
            ))

            bg_series = pd.concat(bg_measurements,axis=0)
            bg_series = bg_series.rename(columns={
                f"{self.background_name}": f"{self.colony_name}",
            })

            def add_bg_name(name):
                split = name.split("_")
                split[1] = f"Background{split[1]}"
                return "_".join(split)

            bg_series.index=bg_series.index.map(lambda x: add_bg_name(x))

            cp_connection.pipeline.end_run()
            validity = pd.DataFrame({
                f"{self.colony_name}": self.status_validity,
            }, index=["status_valid_analysis"]
            ).astype(int)
            self._cp_results = pd.concat([validity, *object_measurements, bg_series], axis=0)
        else:
            log.info(f"Did not analyze {self.sample_name} because of invalid status")


class CellProfilerApiConnection:
    _cp_api_connection = cp_connection

    def refresh(self):
        self._cp_api_connection.refresh()
