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
from typing import List

from ..normalization.plate_normalization import PlateNormalization
from .colony_profile import ColonyProfile, CellProfilerApiConnection

METADATA_LABELS = [
    (STATUS_VALIDITY_LABEL := "status_valid_analysis"),
    (ORIGIN_PLATE_ID_LABEL := "origin_plate_id")
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
    def __init__(self, img: np.ndarray, sample_name: str,
                 sampling_day: int = np.nan,
                 n_rows=8, n_cols=12,
                 align=True, fit=True, use_boost=True,
                 auto_analyze: bool = False,
                 **kwargs
                 ):

        self.__sample_name = sample_name
        self.wells = []
        self.measurement_results = None
        self.sampling_day = sampling_day

        self.cp_connnection = CellProfilerApiConnection()
        self.status_well_analysis = False

        super().__init__(img=img, n_rows=n_rows, n_cols=n_cols,
                         align=align, fit=fit, use_boost=use_boost, auto_run=auto_analyze,
                         **kwargs)

        if auto_analyze:
            self.generate_well_profiles()

    @property
    def sample_name(self):
        return self.__sample_name

    @property
    def results(self):
        """
        Convenience function
        :return: self.get_results()
        """
        return self.get_results()

    def generate_well_profiles(self):
        if self.status_well_analysis is False:
            well_imgs = self.get_well_imgs()
            self.cp_connnection.refresh()

            self.measurement_results = []
            for idx, well_img in enumerate(well_imgs):
                log.debug(f"Starting well analysis for {self.sample_name}_well({idx:03d})")
                tmp_well_name = f"{self.sample_name}_well({idx:03d})"
                well_profile = ColonyProfile(
                        well_img, tmp_well_name, auto_run=True
                )
                self.measurement_results.append(well_profile.get_results())
                self.wells.append(well_profile)
            self.measurement_results = pd.concat(self.measurement_results, axis=1)
            self.measurement_results = self.measurement_results.loc[:, ~self.measurement_results.columns.duplicated()]

            self.status_well_analysis = True
            log.info("Finished generating well profiles")
        else:
            log.info("Status well analysis already generated")

    def get_results(self, numeric_only=False, include_adv: bool = False):
        """
        Returns the results of the CellProfiler Analysis from the API. The results have the measurements row-wise,
        and the plate colonies column-wise.
        :param include_adv:
        :return:
        """
        if self.status_well_analysis is False:
            self.generate_well_profiles()

        if include_adv:
            return self._results
        else:
            measurement_labels = NUMERIC_METADATA_LABELS
            if numeric_only is False:
                measurement_labels = METADATA_LABELS + measurement_labels

            measurement_table = self._results.loc[measurement_labels, :]

            basic_table = []
            for metric in BASIC_CP_API_MEASUREMENT_LABELS:
                basic_table.append(
                        self._results.loc[self._results.index.to_series().str.contains(metric), :]
                )

            results = pd.concat([measurement_table, *basic_table], axis=0).transpose()
            if numeric_only is False:
                results.loc[:, f"{STATUS_VALIDITY_LABEL}"] = results.loc[:, f"{STATUS_VALIDITY_LABEL}"].astype(bool)

            return results

    def add_sampling_day(self, sampling_day: int):
        self.sampling_day = sampling_day

    def get_valid_count(self):
        return self.measurement_results.loc[:, [f"{STATUS_VALIDITY_LABEL}"]].value_counts()

    def set_invalid_well(self, well_idxs: List[int]):
        for well_idx in well_idxs:
            self.wells[well_idx].status_validity = False

    @property
    def _metadata(self):
        metadata = []
        col_idx = self.measurement_results.columns

        day = np.full(shape=len(col_idx),
                      fill_value=self.sampling_day)
        day = pd.DataFrame({
            f"{SAMPLING_DAY_LABEL}": day
        }, index=col_idx).transpose()
        metadata.append(day)

        origin_plate_id = np.full(shape=len(col_idx),
                                  fill_value=self.sample_name)
        origin_plate_id = pd.DataFrame({
            f"{ORIGIN_PLATE_ID_LABEL}": origin_plate_id
        }, index=col_idx).transpose()
        metadata.append(origin_plate_id)

        return pd.concat(metadata, axis=0)

    @property
    def _results(self):
        if self.status_well_analysis is False:
            self.generate_well_profiles()
        results = pd.concat([self._metadata, self.measurement_results], axis=0)
        return results
