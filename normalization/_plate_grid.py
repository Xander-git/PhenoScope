# ----- Imports -----
import numpy as np
import matplotlib.pyplot as plt
from typing import List

import os
import logging
logger_name = "phenomics-normalization"
log = logging.getLogger(logger_name)
logging.basicConfig(format=f'[%(asctime)s|%(levelname)s|{os.path.basename(__file__)}] %(message)s')
# ----- Pkg Relative Import -----
from ._plate_fit import PlateFit
from ..util.plotting import plot_plate_rows


# ----- Main Class Definition -----


class PlateGrid(PlateFit):
    '''
    Last Updated: 7/15/2024
    '''
    cols_midpoints = rows_midpoints = None
    gridded_fig = gridded_ax = None
    invalid_wells = []

    # Status checks
    status_midpoints = False

    def __init__(self, img, n_rows=8, n_cols=12,
                 align=True, fit=True):
        super().__init__(
            img, n_rows, n_cols,
            align, fit
        )

    def run(self):
        super().run()
        try:
            self.find_midpoints()
        except:
            log.warning("Could not find midpoints of image blobs")
            self.status_validity = False
            self._invalid_op = "Invalid: Finding Midpoints"
            self._invalid_op_img = self.img
            self._invalid_blobs = self.blobs

    def find_midpoints(self):
        log.info("Starting calculation for blob edge midpoints")
        rows = self.blobs.rows
        cols = self.blobs.cols

        rows_yMinus = []
        rows_yPlus = []
        for row in rows:
            rows_yMinus.append(row.y_minus.min())
            rows_yPlus.append(row.y_plus.max())
        rows_yMinus = np.array(rows_yMinus)
        rows_yPlus = np.array(rows_yPlus)

        cols_xMinus = []
        cols_xPlus = []
        for col in cols:
            cols_xMinus.append(col.x_minus.min())
            cols_xPlus.append(col.x_plus.max())
        cols_xMinus = np.array(cols_xMinus)
        cols_xPlus = np.array(cols_xPlus)

        self.rows_midpoints = ((rows_yMinus[1:] - rows_yPlus[:-1])/2) + rows_yPlus[:-1]
        self.cols_midpoints = ((cols_xMinus[1:] - cols_xPlus[:-1])/2) + cols_xPlus[:-1]
        self.status_midpoints = True

    def get_well_imgs(self):
        if self.status_midpoints is False:
            self.find_midpoints()
        log.info(f"Getting well images from plate")
        well_imgs = []

        y_start = np.insert(self.rows_midpoints, 0, 0).round().astype(int)
        x_start = np.insert(self.cols_midpoints, 0, 0).round().astype(int)
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

    def plot_well_grid(self, figsize=(12, 8)):
        if self.status_midpoints is False:
            self.find_midpoints()
        with plt.ioff():
            fig, ax = plt.subplots(figsize=figsize)
            ax = self.plotAx_well_grid(ax)
        return fig, ax

    def plotAx_well_grid(self, ax: plt.Axes, fontsize=24):
        if self.status_midpoints is False:
            self.find_midpoints()
        with plt.ioff():
            plot_plate_rows(self.img, self.blobs, ax, set_axis=True)
            for point in self.rows_midpoints:
                ax.axhline(point, linestyle="--")
            for point in self.cols_midpoints:
                ax.axvline(point, linestyle='--')
            ax.set_title("Well Isolation Grid", fontsize=fontsize)
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
