import pandas as pd
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from typing import List

from ..normalization.plate_normalize import PlateNormalize
from .well_profile import WellProfile

import os
import logging

logger_name = "phenomics-phenotyping"
log = logging.getLogger(logger_name)
logging.basicConfig(format=f'[%(asctime)s|%(levelname)s|{os.path.basename(__file__)}] %(message)s')


class PlateProfile(PlateNormalize):
    sample_name = None
    wells = []
    bad_well_imgs = []
    results = None
    gray_imgs = []
    segmentation_imgs = []
    unfiltered_imgs = []
    well_validity = []

    n_rows = n_cols = None

    status_analysis = False
    status_experimental_day = False

    # TODO: Change plate to be an image instead and have plate be generated from the image
    def __init__(self, img, sample_name, n_rows=8, n_cols=12,
                 align=True, fit=True,
                 auto_analyze: bool = False
                 ):
        self.sample_name = sample_name
        super().__init__(img, n_rows, n_cols,
                         align, fit, auto_analyze)
        if auto_analyze:
            self.generate_well_profiles()

    def generate_well_profiles(self):
        well_imgs = self.get_well_imgs()
        results = []
        for idx, well_img in enumerate(well_imgs):
            try:
                log.debug(f"Starting well analysis for {self.sample_name}_well({idx:03d})")
                well_profile = WellProfile(
                    well_img, f"{self.sample_name}_well({idx:03d})", auto_run=True)
                self.wells.append(well_profile)
                results.append(well_profile.get_results())
                self.gray_imgs.append(well_profile.gray_img)
                self.segmentation_imgs.append(well_profile.segmentation)
                self.unfiltered_imgs.append(well_profile.unfiltered_segmentation)
                self.well_validity.append(well_profile.status_validity)
            except:
                self.wells.append(None)
                log.warning(f"Could not analyze well {idx}", exc_info=True)
                self.bad_well_imgs.append(idx)
        self.results = pd.concat(results, axis=1)
        self.results = self.results.loc[:, ~self.results.columns.duplicated()].copy()
        self.status_analysis = True
        log.info("Finished generating well profiles")

    def get_results(self, include_adv: bool = False):
        if self.status_analysis is False:
            self.generate_well_profiles()
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

        basic_list = ["status_validity"] + basic_list

        if include_adv:
            return self.results.copy()
        else:
            basic_table = [self.results.loc[basic_list, :]]
            texture_table_set = []
            for metric in texture_list:
                texture_table_set.append(self.results[self.results.index.to_series().str.contains(metric)])
            return pd.concat(basic_table + texture_table_set, axis=0).copy()

    def plot_analysis(self):
        with plt.ioff():
            fig, axes = plt.subplots(nrows=self.n_rows, ncols=self.n_cols, figsize=(16, 12),
                                     sharex=True, sharey=True)
            for idx, ax in enumerate(axes.ravel()):
                if self.well_validity[idx]:
                    ax.imshow(self.segmentation_imgs[idx], cmap="viridis")
                else:
                    ax.imshow(self.gray_imgs[idx], cmap="gray")
                ax.set_axis_off()
                ax.set_title(f"Well {idx}")
            fig.suptitle(self.sample_name)
        return fig, axes

    def plot_unfiltered(self):
        with plt.ioff():
            fig, axes = plt.subplots(nrows=self.n_rows, ncols=self.n_cols, figsize=(16, 12),
                                     sharex=True, sharey=True)
            for idx, ax in enumerate(axes.ravel()):
                if self.well_validity[idx]:
                    ax.imshow(self.unfiltered_imgs[idx], cmap="viridis")
                else:
                    ax.imshow(self.gray_imgs[idx], cmap="gray")
                ax.set_axis_off()
                ax.set_title(f"Well {idx}")
        return fig, axes

    def add_sampling_day(self, sampling_day: int):
        day = np.full(len(self.results.index), sampling_day)
        self.results.insert(0, "sampling_day", day)
        self.status_experimental_day = True

    def get_validity(self):
        return self.results.loc[["status_validity"], :].transpose().value_counts()
