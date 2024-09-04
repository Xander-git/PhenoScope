import numpy as np

from skimage.color import label2rgb

from ._object_finder_bins import ObjectFinderBins
from ..util.image_analysis import view_img_info


class ObjectFinderView(ObjectFinderBins):
    @property
    def row_map(self):
        r_map = np.zeros(shape=self._obj_map.shape)
        for idx, row_table in enumerate(self.rows):
            row_labels = row_table.index.to_numpy()
            r_map[np.isin(self._obj_map, row_labels)] = idx + 1
        return r_map

    @property
    def col_map(self):
        c_map = np.zeros(shape=self._obj_map.shape)
        for idx, col_table in enumerate(self.cols):
            col_labels = col_table.index.to_numpy()
            c_map[np.isin(self._table, col_labels)] = idx + 1
        return c_map

    def get_row_img(self):
        return label2rgb(label=self.row_map, image=self.input_img)

    def get_col_img(self):
        return label2rgb(label=self.col_map, image=self.input_img)

    def view_row_img(self):
        fig, axes = view_img_info(self.input_img)
        ax = axes[0]
        ax.clear()
        ax.imshow(self.get_row_img())
        return fig, axes

    def view_col_img(self):
        fig, axes = view_img_info(self.input_img)
        ax = axes[0]
        ax.clear()
        ax.imshow(self.get_col_img())
        return fig, axes
