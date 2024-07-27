from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from ..cellprofiler_api.cp_api_analysis import CellProfilerApiAnalysis

import os
import logging
logger_name = "phenomics-phenotyping"
log = logging.getLogger(logger_name)
logging.basicConfig(format=f'[%(asctime)s|%(levelname)s|{os.path.basename(__file__)}] %(message)s')

class WellProfile:
    input_img = None
    gray_img = None
    segmentation = None
    unfiltered_segmentation = None
    results = None

    well_name = None
    primary_name = None
    merged_name = None
    filled_name = None
    colony_name = None
    noise_name = None

    cp_connection = None
    status_validity = None


    def __init__(self, img, sample_name,
                 auto_run=True):
        self.input_img = img
        self.gray_img = ski.color.rgb2gray(img)

        self._set_name(sample_name)
        if auto_run:
            self.run_analysis()

    def run_analysis(self):
        cp_connection = CellProfilerApiAnalysis(self.input_img, self.well_name)
        results = cp_connection.run()
        self.results = results.results
        self.segmentation = results.segmentation
        self.unfiltered_segmentation = results.unfiltered_segmentation
        self.status_validity = deepcopy(results.status_validity)

    def get_results(self):
        return self.results

    def _set_name(self, well_name):
        self.well_name = well_name
        self.primary_name = f"{well_name}_PrimaryObjects"
        self.merged_name = f"{well_name}_MergedObjects"
        self.filled_name = f"{well_name}_FilledObjects"
        self.noise_name = f"{well_name}_Noise"
        self.colony_name = f"{well_name}_Colony"

    def plotAx_colony(self, ax, cmap="viridis"):
        img = self.input_img[self.segmentation > 0, :]
        with plt.ioff():
            ax.imshow(img, cmap=cmap)
            ax.set_title("Segmented Colony")

    def plot_colony(self, cmap="viridis"):
        with plt.ioff():
            fig, ax = plt.subplots()
            self.plotAx_colony(ax, cmap)
        return fig, ax

    def plotAx_segmentation(self, ax, cmap="viridis"):
        with plt.ioff():
            if self.status_validity:
                ax.imshow(self.segmentation, cmap=cmap)
                ax.set_title("Segmented Colony")
            else:
                ax.imshow(self.unfiltered_segmentation, cmap="YlOrRd")
        return ax

    def plotAx_unfiltered(self, ax, unfiltered_img, gray_img ,cmap="viridis"):
        with plt.ioff():
            if self.status_validity:
                ax.imshow(unfiltered_img, cmap=cmap)
                ax.set_title("Segmented Colony")
            else:
                ax.imshow(self.gray_img, cmap="YlOrRd")
        return ax

    def plot_segmentation(self, cmap='viridis'):
        with plt.ioff():
            fig, ax = plt.subplots()
            self.plotAx_segmentation(ax, cmap)
        return fig, ax