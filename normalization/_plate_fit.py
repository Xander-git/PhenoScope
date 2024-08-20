# ----- Imports -----
import numpy as np
import math
import matplotlib.pyplot as plt

import os
import logging

logger_name = "phenomics-normalization"
log = logging.getLogger(logger_name)
logging.basicConfig(format=f'[%(asctime)s|%(levelname)s|{os.path.basename(__file__)}] %(message)s')
# ----- Pkg Relative Import -----
from ._plate_align import PlateAlignment
from ..util.plotting import plot_blobs


# ----- Main Class Definition -----


class PlateFit(PlateAlignment):
    border_padding = 50
    edge_correction_sz = 10
    padded_img = cropping_rect = None

    run_fitting = True
    run_edge_correction = True

    status_fitted = False
    status_pad_img = False

    def __init__(self, img, n_rows=8, n_cols=12, align=True, fit=True):
        super().__init__(img, n_rows, n_cols, align)
        self.run_fitting = fit

    def run(self):
        try:
            if self.run_fitting:
                padded_img = []
                for color_channel in range(3):
                    tmp_img = self.img[:, :, color_channel]

                    padded_img.append(
                        np.expand_dims(
                            np.pad(tmp_img, self.border_padding, mode='edge'),
                            axis=2
                        )
                    )
                self.padded_img = np.concatenate(padded_img, axis=2)
                self.input_img = self.padded_img
                self._set_img(self.padded_img)
                super().run()
                self.fit_plate()
            else:
                super().run()
        except:
            log.warning("Could not fit image")
            self.status_validity = False
            self._invalid_op = "Invalid: Fitting"
            self._invalid_op_img = self.img
            self._invalid_blobs = self.blobs

    def fit_plate(self):
        if self.status_alignment is False:
            self.align()
        log.info("Starting plate fitting")
        bound_L = math.floor(self.blobs.cols[0].x_minus.min() - self.border_padding)
        bound_R = math.ceil(self.blobs.cols[-1].x_plus.max() + self.border_padding)
        bound_T = math.floor(self.blobs.rows[0].y_minus.min() - self.border_padding)
        bound_B = math.ceil(self.blobs.rows[-1].y_plus.max() + self.border_padding)
        self._set_img(self.img[bound_T:bound_B, bound_L:bound_R])
        with plt.ioff():
            self.cropping_rect = plt.Rectangle((bound_L, bound_T),
                                               bound_R - bound_L,
                                               bound_B - bound_T,
                                               fill=False, edgecolor='white')
        log.info("Updating blobs after fitting")
        self._update_blobs()
        self.status_fitted = True

    def plot_fitting(self):
        if self.status_alignment is False:
            self.align()
        if self.status_fitted is False:
            self.fit_plate()
        with plt.ioff():
            alignFit_fig, alignFit_ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 10),
                                                     tight_layout=True)
            self.plotAx_fitting(alignFit_ax[0])

            alignFit_ax[1].imshow(self.img)
            alignFit_ax[1].set_title("Fitted Image")
            for idx, row in self.blobs.table.iterrows():
                c = plt.Circle((row['x'], row['y']), row['radius'], color='green', fill=False)
                alignFit_ax[1].add_patch(c)
            alignFit_ax[1].grid(False)
        return alignFit_fig, alignFit_ax

    def plotAx_fitting(self, ax: plt.Axes, fontsize=24):
        if self.status_alignment is False:
            self.align()
        if self.status_fitted is False:
            self.fit_plate()

        with plt.ioff():
            ax.grid(False)
            ax.imshow(self.padded_img)
            ax.add_patch(self.cropping_rect)
            ax.set_title("Cropping Outline", fontsize=fontsize)
        return ax

    def get_fitted_blob_plot(self):
        with plt.ioff():
            if self.status_fitted is False:
                self.fit_plate()

            fig, ax = plot_blobs(self.img, self.blobs.table, grayscale=False)
        return fig, ax
