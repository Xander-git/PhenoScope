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
            try:
                results = cp_connection.run(self)
                self._cp_results = results.results
                self.status_validity = deepcopy(results.status_validity)
                self.status_analysis = True
            except KeyboardInterrupt:
                sys.exit("User exited with keyboard interrupt\n")
            except:
                log.warning(f"Failed to analyze {self.sample_name}")
                self.status_validity = False
        else:
            log.info(f"Did not analyze {self.sample_name} because of invalid status")


class CellProfilerApiConnection:
    _cp_api_connection = cp_connection

    def refresh(self):
        self._cp_api_connection.refresh()
