import pandas as pd
import numpy as np

from ..normalization.plate_normalization import PlateNormalization
from .colony_profile import ColonyProfile

import logging

log = logging.getLogger(__file__)
logging.basicConfig(format=f'[%(asctime)s|%(levelname)s|%(name)s] %(message)s')


class PlateProfileBase(PlateNormalization):
    # TODO: Change plate to be an image instead and have plate be generated from the image
    def __init__(self, img, sample_name, sampling_day=None, n_rows=8, n_cols=12,
                 align=True, fit=True,
                 auto_analyze: bool = False
                 ):
        super().__init__(img, n_rows, n_cols,
                         align, fit, auto_analyze)
        self._sample_name = (sample_name, None)
        self.wells = []
        self.well_names = []
        self.bad_well_img_idxs = []
        self.raw_results = None

        self.status_analysis = False
        self.status_collect_and_clean_object_data = False
        self.status_sampling_day = False

        if sampling_day is not None:
            self.sampling_day = sampling_day

        if auto_analyze:
            self.generate_well_profiles()

    @property
    def sample_name(self):
        return self._sample_name[0]

    def generate_well_profiles(self):
        well_imgs = self.get_well_imgs()
        results = []
        for idx, well_img in enumerate(well_imgs):
            try:
                log.debug(f"Starting well analysis for {self.sample_name}_well({idx:03d})")
                tmp_well_name = f"{self.sample_name}_well({idx:03d})"
                self.well_names.append(tmp_well_name)
                well_profile = ColonyProfile(
                    well_img, tmp_well_name, auto_run=True)
                self.wells.append(well_profile)
                results.append(well_profile.get_results())
            except:
                self.wells.append(None)
                log.warning(f"Could not analyze well {idx}", exc_info=True)
                self.bad_well_img_idxs.append(idx)
        # self._collect_and_clean_object_data()
        self.well_names = pd.Series(self.well_names)

        self.raw_results = pd.concat(results, axis=1)
        self.raw_results = self.raw_results.loc[:, ~self.raw_results.columns.duplicated()]
        if self.sampling_day is not None:
            self.add_sampling_day(self.sampling_day)
        self.status_analysis = True
        log.info("Finished generating well profiles")

    def get_results(self, include_adv: bool = False):
        """
        Returns the results of the CellProfiler Analysis from the API. The results have the measurements row-wise,
        and the plate colonies column-wise.
        :param include_adv:
        :return:
        """
        basic_list = [
            "AreaShape_Area",
            "AreaShape_Perimeter",
            "AreaShape_FormFactor",
            "AreaShape_Compactness",
            "AreaShape_Extent",
            "AreaShape_Eccentricity",
            "AreaShape_MaximumRadius",
            "AreaShape_Orientation",
            "Intensity_IntegratedIntensity",
            "Intensity_MeanIntensity",
            "Intensity_MaxIntensity",
            "Intensity_MinIntensity",
        ]

        texture_list = [
            "Texture_Contrast",
            "Texture_Correlation",
            "Texture_Variance",
            "Texture_AngularSecondMoment",
            "Texture_Entropy"
        ]

        if self.status_sampling_day is False:
            basic_list = ["status_valid_analysis"] + basic_list
        else:
            basic_list = ["status_valid_analysis", "sampling_day"] + basic_list

        if include_adv:
            return self._results
        else:
            basic_table = [self._results.loc[basic_list, :]]
            texture_table_set = []
            for metric in texture_list:
                texture_table_set.append(
                    self._results[self._results.index.to_series().str.contains(metric)]
                )
            return pd.concat(basic_table + texture_table_set, axis=0).copy()

    def validate_well_names(self):
        """
        Resets the well names to their intended names. To be used before plotting. This is needed because cellprofiler sometimes changes the naming due to how it is processed
        :return:
        """
        self.well_names = self.well_names[
            self.well_names.str.contains(f"{self.sample_name}", regex=False)
        ]
        for idx, name in enumerate(self.well_names):
            if self.wells[idx] is not None:
                self.wells[idx]._well_name = (name, None)

    def add_sampling_day(self, sampling_day: int):
        if self.status_sampling_day is False:
            self.sampling_day = sampling_day
            day = np.full(len(self.raw_results.columns), sampling_day)
            day = pd.DataFrame({
                "sampling_day": day
            }, index=self.raw_results.columns).transpose()
            self.raw_results = pd.concat([self.raw_results, day], axis=0)
            self.status_sampling_day = True

    def get_valid_count(self):
        return self.raw_results.loc[:, ["status_valid_analysis"]].value_counts()

    @property
    def _results(self):
        if self.status_analysis is False:
            self.generate_well_profiles()
        results = self.raw_results.copy()
        return results

    def _collect_and_clean_object_data(self):
        # TODO: Fix this function and add removal of collected variables
        log.info("Collecting and cleaning well object data from CellProfiler API")
        for well in self.wells:
            well._pull_object_info()
        self.status_collect_and_clean_object_data = True
