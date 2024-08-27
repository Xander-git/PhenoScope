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

from ..normalization.plate_normalization import PlateNormalization
from .colony_profile import ColonyProfile, CellProfilerApiConnection

METADATA_LABELS = [
    (STATUS_VALIDITY_LABEL := "status_valid_analysis"),
    (PLATE_NAME_LABEL := "plate_name")
]

NUMERIC_METADATA_LABELS = [
    (SAMPLING_DAY_LABEL := "sampling_day"),
]

PHENOMICS_MEASUREMENTS = [
    "Intensity_IntegratedColorIntensityRed",
    "Intensity_IntegratedColorIntensityGreen",
    "Intensity_IntegratedColorIntensityBlue",
    "ImgHeight",
    "ImgWidth"
]

BASIC_CP_API_MEASUREMENT_LABELS = [
    "AreaShape",
    "Intensity",
    "Texture"
]


# ----- Main Class Definition -----
class PlateProfileBase(PlateNormalization):
    # TODO: Change plate to be an image instead and have plate be generated from the image
    def __init__(self, img: np.ndarray, plate_name: str,
                 sampling_day: int = np.nan,
                 n_rows: int = 8, n_cols: int = 12,
                 use_boost: bool = True,
                 **kwargs
                 ):

        self.__plate_name = plate_name
        self.wells = []
        self.measurement_results = None
        self._sampling_day = sampling_day

        self.cp_connnection = CellProfilerApiConnection()
        self.status_well_analysis = False
        self.status_measure = False

        super().__init__(img=img,
                         n_rows=n_rows,
                         n_cols=n_cols,
                         use_boost=use_boost,
                         **kwargs)

    @property
    def plate_name(self):
        return self.__plate_name

    @property
    def results(self):
        """
        Convenience function
        :return: pd.DataFrame
        """
        return self.get_results()

    def run(self, align=True, fit=True):
        super().run(align=align, fit=fit)
        if self.status_measure is False: self.measure()

    def measure(self):
        self._measure()
        self.status_measure = True

    def _measure(self):
        self._generate_well_profiles()

    def get_results(self, numeric_only: bool = False, include_adv: bool = False):
        """
        Returns the results of the CellProfiler Analysis from the API. The results have the measurements row-wise,
        and the plate colonies column-wise.
        :param numeric_only: whether to return only the numer results or not
        :param include_adv: whether to include the advanced measurements taken by cellprofiler as well
        :return:
        """
        if self.status_well_analysis is False:
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

    def set_sampling_day(self, sampling_day: int):
        self._sampling_day = sampling_day

    def get_valid_count(self):
        return self.measurement_results.loc[:, [f"{STATUS_VALIDITY_LABEL}"]].value_counts()

    def set_invalid_well(self, well_idxs: List[int]):
        for well_idx in well_idxs:
            self.wells[well_idx].status_validity = False

    def _generate_well_profiles(self):
        if self.status_well_analysis is False:
            well_imgs = self.get_well_imgs()
            self.cp_connnection.refresh()

            self.measurement_results = []
            for idx, well_img in enumerate(well_imgs):
                try:
                    log.debug(f"Starting well analysis for {self.plate_name}_well({idx:03d})")
                    tmp_well_name = f"well({idx:03d})"
                    well_profile = ColonyProfile(
                            well_img, tmp_well_name, auto_run=True
                    )
                    self.measurement_results.append(well_profile.get_results())
                    self.wells.append(well_profile)
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except:
                    log.warning(f"Failed to analyze well {idx} for plate {self.plate_name}")
            self.measurement_results = pd.concat(self.measurement_results, axis=1)
            self.measurement_results = self.measurement_results.loc[:, ~self.measurement_results.columns.duplicated()]

            self.status_well_analysis = True
            log.info("Finished generating well profiles")
        else:
            log.info("Status well analysis already generated")

    @property
    def _metadata(self):
        metadata = []
        col_idx = self.measurement_results.columns

        day = np.full(shape=len(col_idx),
                      fill_value=self._sampling_day)
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
        results = pd.concat([self._metadata, self.measurement_results], axis=0)
        return results
