# ----- Imports -----
import numpy as np
import matplotlib.pyplot as plt
from typing import List

import os
import logging

logger_name = "phenomics-preprocessing"
log = logging.getLogger(logger_name)
logging.basicConfig(format=f'[%(asctime)s|%(levelname)s|{os.path.basename(__file__)}] %(message)s')
# ----- Pkg Relative Import -----
from ._plate_fit import PlateFit
from phenoscope.util.plotting import plot_plate_rows


# ----- Main Class Definition -----
class PlateGrid(PlateFit):
    def __init__(
            self, img: np.ndarray,
            n_rows: int = 8,
            n_cols: int = 12,
            border_padding: int = 50,
            **kwargs
    ):

        self.cols_midpoints = None
        self.rows_midpoints = None
        self.gridded_fig = None
        self.gridded_ax = None

        self._status_midpoints = False

        super().__init__(
                img=img,
                n_rows=n_rows,
                n_cols=n_cols,
                border_padding=border_padding,
                **kwargs
        )

    def run(self, align=True, fit=True):
        self.normalize(align=align, fit=fit)

    def _normalize(self, align=True, fit=True):
        super()._normalize(align=align, fit=fit)
        self.find_midpoints()

    def find_midpoints(self):
        if self.blobs.empty: self._update_blobs()
        if self.blobs.empty is True: raise ValueError(
                "No blobs were found in the image. Try increasing the contrast of the image or change preprocessing parameters"
        )

        log.info("Starting calculation for blob edge midpoints")
        rows = self.blobs.gridrows
        cols = self.blobs.gridcols

        rows_yMinus = []
        rows_yPlus = []
        for row in rows:
            rows_yMinus.append(row.loc[:, 'y_minus'].min())
            rows_yPlus.append(row.loc[:, 'y_plus'].max())
        rows_yMinus = np.array(rows_yMinus)
        rows_yPlus = np.array(rows_yPlus)

        cols_xMinus = []
        cols_xPlus = []
        for col in cols:
            cols_xMinus.append(col.x_minus.min())
            cols_xPlus.append(col.x_plus.max())
        cols_xMinus = np.array(cols_xMinus)
        cols_xPlus = np.array(cols_xPlus)

        self.rows_midpoints = ((rows_yMinus[1:] - rows_yPlus[:-1]) / 2) + rows_yPlus[:-1]
        self.cols_midpoints = ((cols_xMinus[1:] - cols_xPlus[:-1]) / 2) + cols_xPlus[:-1]
        self._status_midpoints = True

    def get_well_imgs(self):
        if self._status_midpoints is False: self.find_midpoints()
        log.info(f"Getting well images from plate")
        well_imgs = []

        y_start = np.insert(self.rows_midpoints, obj=0, values=0).round().astype(int)
        x_start = np.insert(self.cols_midpoints, obj=0, values=0).round().astype(int)
        y_end = np.append(self.rows_midpoints, self.img.shape[0] - 1).round().astype(int)
        x_end = np.append(self.cols_midpoints, self.img.shape[1] - 1).round().astype(int)

        for row_idx in range(self.n_rows):
            for col_idx in range(self.n_cols):
                well_imgs.append(
                        self.img[
                        y_start[row_idx]:y_end[row_idx],
                        x_start[col_idx]:x_end[col_idx]
                        ]
                )
        return well_imgs

    def plot_plate_gridding(self, figsize=(12, 8)):
        if self._status_midpoints is False:
            self.find_midpoints()
        with plt.ioff():
            fig, ax = plt.subplots(figsize=figsize)
            ax = self.plotAx_plate_gridding(ax)
        return fig, ax

    def plotAx_plate_gridding(self, ax: plt.Axes, fontsize=24):
        if self._status_midpoints is False:
            self.find_midpoints()
        with plt.ioff():
            plot_plate_rows(self.img, self.blobs, ax, set_axis=True)
            for point in self.rows_midpoints:
                ax.axhline(point, linestyle="--")
            for point in self.cols_midpoints:
                ax.axvline(point, linestyle='--')
            ax.set_title("Plate Gridding", fontsize=fontsize)
        return ax

    def plot_well_imgs(self, figsize=(20, 18)):
        well_imgs = self.get_well_imgs()
        with plt.ioff():
            wells_fig, wells_ax = plt.subplots(nrows=self.n_rows, ncols=self.n_cols, tight_layout=True, figsize=figsize)
            for idx, ax in enumerate(wells_ax.ravel()):
                ax.set_title(f"sample_idx: {idx}")
                ax.imshow(well_imgs[idx])
                ax.set_axis_off()
        return wells_fig, wells_ax
