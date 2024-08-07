from copy import deepcopy
import matplotlib.pyplot as plt
import skimage as ski
from ..cellprofiler_api.cell_profiler_api import CellProfilerApi

import logging
log = logging.getLogger(__file__)
logging.basicConfig(format=f'[%(asctime)s|%(levelname)s|%(name)s] %(message)s')

cp_connection = CellProfilerApi()
class ColonyProfile:
    def __init__(self, img, sample_name,
                 auto_run=True):
        self._input_img = (img, None)
        self._sample_name = (sample_name, None)
        self.status_validity = False
        self.segmentation = None
        self.unfiltered_segmentation = None
        if auto_run:
            self.run_analysis()

    @property
    def input_img(self):
        return self._input_img[0]

    @property
    def sample_name(self):
        return self._sample_name[0]

    @property
    def gray_img(self):
        return ski.color.rgb2gray(self.input_img)

    @property
    def primary_name(self):
        return f"{self.sample_name}_PrimaryObjects"

    @property
    def merged_name(self):
        return f"{self.sample_name}_MergedObjects"

    @property
    def filled_name(self):
        return f"{self.sample_name}_FilledObjects"

    @property
    def noise_name(self):
        return f"{self.sample_name}_Noise"

    @property
    def colony_name(self):
        return f"{self.sample_name}_Colony"

    def run_analysis(self):
        results = cp_connection.run(self.input_img, f"{self.sample_name}")
        self.results = results.results
        self.status_validity = deepcopy(results.status_validity)

    def get_results(self):
        return self.results.copy()

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
                # ax.imshow(self.segmentation, cmap=cmap)
                ax.imshow(
                    cp_connection.obj_set.get_objects(f"{self.colony_name}").segmented,
                    cmap=cmap
                )
                ax.set_title("Segmented Colony")
            else:
                ax.imshow(self.gray_img, cmap="YlOrRd")
        return ax

    def plotAx_unfiltered(self, ax,cmap="viridis"):
        with plt.ioff():
            if self.status_validity:
                # ax.imshow(self.unfiltered_segmentation, cmap=cmap)
                ax.imshow(
                    cp_connection.obj_set.get_objects(f"{self.filled_name}").segmented,
                    cmap=cmap
                )
                ax.set_title("Segmented Colony")
            else:
                ax.imshow(self.gray_img, cmap="YlOrRd")
        return ax

    def plot_segmentation(self, cmap='viridis'):
        with plt.ioff():
            fig, ax = plt.subplots()
            self.plotAx_segmentation(ax, cmap)
        return fig, ax

    def _pull_object_info(self):
        self.segmentation = cp_connection.obj_set.get_objects(self.colony_name).segmented
        self.unfiltered_segmentation = cp_connection.obj_set.get_objects(self.filled_name).segmented
