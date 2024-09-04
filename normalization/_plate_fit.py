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
    def __init__(self, img: np.ndarray,
                 n_rows: int = 8, n_cols: int = 12,
                 border_padding: int = 50,
                 **kwargs
                 ):
        self.border_padding = border_padding
        self.padded_img = None
        self.cropping_rect = None

        self._status_fitted = False
        self._status_pad_img = False
        super().__init__(img=img,
                         n_rows=n_rows,
                         n_cols=n_cols, **kwargs)

    def normalize(self, align: bool = True, fit: bool = True):
        self._normalize(align=align, fit=fit)
        self._status_normalization = True

    def _normalize(self, align: bool = True, fit: bool = True):
        if fit and not self._status_pad_img: self._pad_img()
        super()._normalize(align=align)
        if fit and not self._status_pad_img: self.fit_plate()

    def _pad_img(self):
        self.padded_img = []
        for color_channel in range(3):
            tmp_img = self.img[:, :, color_channel]
            self.padded_img.append(
                    np.expand_dims(
                            np.pad(
                                    array=tmp_img,
                                    pad_width=self.border_padding,
                                    mode='edge'
                            ), axis=2
                    )
            )
        self.padded_img = np.concatenate(self.padded_img, axis=2)
        self._set_img(self.padded_img)

    def fit_plate(self):
        if self.blobs.empty: self._update_blobs()
        if self.blobs.empty is True: raise ValueError(
                "No blobs were found in the image. Try increasing the contrast of the image or change preprocessing parameters"
        )

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
        self._status_fitted = True

    def plot_fitting(self):
        if self._status_alignment is False:
            self.align()
        if self._status_fitted is False:
            self.fit_plate()
        with plt.ioff():
            alignFit_fig, alignFit_ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 10),
                                                     tight_layout=True)
            self.plotAx_fitting(alignFit_ax[0])

            alignFit_ax[1].imshow(self.img)
            alignFit_ax[1].set_title("Fitted Image")
            for idx, row in self.blobs.results.iterrows():
                c = plt.Circle((row['x'], row['y']), row['radius'], color='green', fill=False)
                alignFit_ax[1].add_patch(c)
            alignFit_ax[1].grid(False)
        return alignFit_fig, alignFit_ax

    def plotAx_fitting(self, ax: plt.Axes, fontsize=24):
        if self._status_alignment is False:
            self.align()
        if self._status_fitted is False:
            self.fit_plate()

        with plt.ioff():
            ax.grid(False)
            ax.imshow(self.padded_img)
            ax.add_patch(self.cropping_rect)
            ax.set_title("Cropping Outline", fontsize=fontsize)
        return ax

    def get_fitted_blob_plot(self):
        with plt.ioff():
            if self._status_fitted is False:
                self.fit_plate()

            fig, ax = plot_blobs(self.img, self.blobs.results, grayscale=False)
        return fig, ax
