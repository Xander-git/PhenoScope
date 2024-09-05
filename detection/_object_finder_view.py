from typing import Tuple, List

import numpy as np

from skimage.color import label2rgb
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from ._object_finder_bins import ObjectFinderBins
from ..util.image_analysis import view_img_info


class ObjectFinderView(ObjectFinderBins):
    @property
    def gridrow_map(self) -> np.ndarray:
        gridrow_map = np.zeros(shape=self._obj_map.shape)
        for r_idx, row_table in enumerate(self.gridrows):
            row_labels = row_table.index.to_numpy()
            gridrow_map[np.isin(element=self._obj_map, test_elements=row_labels)] = r_idx + 1
        return gridrow_map

    @property
    def gridcol_map(self) -> np.ndarray:
        gridcol_map = np.zeros(shape=self._obj_map.shape)
        for c_idx, col_table in enumerate(self.gridcols):
            col_labels = col_table.index.to_numpy()
            gridcol_map[np.isin(element=self._obj_map, test_elements=col_labels)] = c_idx + 1
        return gridcol_map

    def get_gridrow_overlay(self, image: np.ndarray) -> np.ndarray:
        return label2rgb(label=self.gridrow_map, image=image)

    def get_gridcol_overlay(self, image: np.ndarray) -> np.ndarray:
        return label2rgb(label=self.gridcol_map, image=image)

    def view_gridrow_overlay(self, image: np.ndarray) -> Tuple[Figure, Axes]:
        fig, axes = view_img_info(image)
        ax = axes[0]
        ax.clear()
        ax.imshow(self.get_gridrow_overlay(image=image))
        return fig, axes

    def view_gridcol_overlay(self, image: np.ndarray) -> Tuple[Figure, Axes]:
        fig, axes = view_img_info(img=image)
        ax = axes[0]
        ax.clear()
        ax.imshow(self.get_gridcol_overlay(image=image))
        return fig, axes
