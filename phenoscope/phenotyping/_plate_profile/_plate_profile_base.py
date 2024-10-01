# ----- Logger Setup -----
import logging

formatter = logging.Formatter(
        fmt=f'[%(asctime)s|%(name)s] %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S'
)
console_handler = logging.StreamHandler()
log = logging.getLogger(__name__)
log.addHandler(console_handler)
console_handler.setFormatter(formatter)
# ----- Imports -----

import sys
import pandas as pd
import numpy as np
from typing import List

from ...preprocessing.plate_preprocessing import PlatePreprocessing
from ..colony_profile import ColonyProfile, CellProfilerApiConnection

from ...util._labels import (METADATA_LABELS, NUMERIC_METADATA_LABELS, PHENOSCOPE_MEASUREMENTS,
                             BASIC_CP_API_MEASUREMENT_LABELS)
from ...util._labels import SAMPLING_DAY_LABEL, PLATE_NAME_LABEL, STATUS_VALIDITY_LABEL


# ----- Main Class Definition -----
class PlateProfileBase(PlatePreprocessing):
    # TODO: Change plate to be an image instead and have plate be generated from the image
    def __init__(
            self, img: np.ndarray, plate_name: str,
            sampling_day: int = np.nan,
            n_rows: int = 8, n_cols: int = 12,
            use_boost: bool = True,
            **kwargs
    ):

        self.__plate_name = plate_name
        self.wells = []
        self._measurement_results = pd.DataFrame()
        self.__sampling_day = sampling_day

        self.cp_connnection = CellProfilerApiConnection()
        self.status_well_profiles = False
        self.status_measure = False

        super().__init__(img=img,
                         n_rows=n_rows,
                         n_cols=n_cols,
                         use_boost=use_boost,
                         **kwargs)

    @property
    def plate_name(self):
        return self.__plate_name

    def run(self, align=True, fit=True):
        super().run(align=align, fit=fit)
        if self.status_measure is False: self.measure()

    def measure(self):
        if self.status_measure is False:
            self._measure()
            self.status_measure = True
        return self.results

    def _measure(self):
        self._generate_well_profiles()

    @property
    def results(self):
        """
        Convenience function
        :return: pd.DataFrame
        """
        return self.get_results()

    def get_results(self, numeric_only: bool = False, include_adv: bool = False):
        """
        Returns the results of the CellProfiler Analysis from the API. The results have the measurements row-wise,
        and the plate colonies column-wise.
        :param numeric_only: whether to return only the numer results or not
        :param include_adv: whether to include the advanced measurements taken by cellprofiler as well
        :return:
        """
        if self.status_well_profiles is False:
            self._generate_well_profiles()

        if include_adv:
            return self._results
        else:
            metadata = NUMERIC_METADATA_LABELS
            if numeric_only is False:
                metadata = METADATA_LABELS + metadata

            metadata_table = self._results.loc[metadata, :]

            basic_table = []
            for metric in BASIC_CP_API_MEASUREMENT_LABELS:
                basic_table.append(
                        self._results.loc[self._results.index.to_series().str.contains(metric), :]
                )

            results = pd.concat([metadata_table, *basic_table], axis=0).transpose()
            if numeric_only is False:
                results.loc[:, f"{STATUS_VALIDITY_LABEL}"] = results.loc[:, f"{STATUS_VALIDITY_LABEL}"].astype(bool)

            return results

    def set_sampling_day(self, sampling_day: int) -> None:
        self.__sampling_day = sampling_day

    def get_sampling_day(self) -> int:
        return self.__sampling_day

    def get_valid_count(self):
        return self._measurement_results.loc[:, [f"{STATUS_VALIDITY_LABEL}"]].value_counts()

    def set_invalid_well(self, well_idxs: List[int]):
        for well_idx in well_idxs:
            self.wells[well_idx].status_validity = False

    def _generate_well_profiles(self):
        if self.status_well_profiles is False:
            well_imgs = self.get_well_imgs()
            self.cp_connnection.refresh()

            self._measurement_results = []
            for idx, well_img in enumerate(well_imgs):
                log.debug(f"Starting well analysis for {self.plate_name}_well({idx:03d})")

                well_profile = ColonyProfile(
                        img=well_img, image_name=f"well({idx:03d})",
                        auto_run=False,
                        use_boosted_mask=self._use_boost
                )
                try:
                    well_profile.run()

                except KeyboardInterrupt:
                    raise KeyboardInterrupt

                except Exception as e:
                    log.warning(f"Failed to analyze Plate({self.plate_name}_well({idx:03d})) : {e}")
                    well_profile.status_validity = False

                finally:
                    self.wells.append(well_profile)
                    self._measurement_results.append(well_profile.get_results())

            # TODO: See if possibile to remove this layer
            self._measurement_results = pd.concat(self._measurement_results, axis=1)
            self._measurement_results = self._measurement_results.loc[:,
                                        ~self._measurement_results.columns.duplicated()]

            self.status_well_profiles = True
            log.info("Finished generating well profiles")
        else:
            log.info("Status well analysis already generated")

    def get_metadata(self):
        metadata = []
        col_idx = self._measurement_results.columns

        day = np.full(shape=len(col_idx),
                      fill_value=self.__sampling_day)
        day = pd.DataFrame({
            f"{SAMPLING_DAY_LABEL}": day
        }, index=col_idx).transpose()
        metadata.append(day)

        PLATE_NAME_METADATA = np.full(shape=len(col_idx),
                                      fill_value=self.plate_name)
        PLATE_NAME_METADATA = pd.DataFrame({
            f"{PLATE_NAME_LABEL}": PLATE_NAME_METADATA
        }, index=col_idx).transpose()
        metadata.append(PLATE_NAME_METADATA)

        return pd.concat(metadata, axis=0)

    @property
    def _results(self):
        if self.status_measure is False: self.measure()
        results = pd.concat([self.get_metadata(), self._measurement_results], axis=0)
        return results
