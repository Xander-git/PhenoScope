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

from ._colony_profile_plot_object import ColonyProfilePlotObject


class ColonyProfileMeasureBase(ColonyProfilePlotObject):
    def __init__(self, img, sample_name, auto_run=True):
        self._measurements = {}
        super().__init__(img=img, sample_name=sample_name, auto_run=auto_run)

    def run_analysis(self,
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
        if self.status_validity:
            try:
                self._measure_colony()
            except KeyboardInterrupt:
                sys.exit("...User terminated program")
            except:
                log.warning(f"could not measure colony for sample: {self.sample_name}")
                self.status_validity = False
        else:
            log.info(f"Did not measure colony due to invalid analysis: {self.sample_name}")

    def _measure_colony(self):
        """
        Except when changing the exception handling protocol of the module,
        change this class when adding new measurements to maintain intended exception handling protocol
        :return:
        """
        pass
