# ----- Imports -----
import numpy as np
import math
import matplotlib.pyplot as plt

# ----- Pkg Relative Import -----
from .plate_align import PlateAlignment
from ..util.plotting import plot_blobs

# ----- Main Class Definition -----


class PlateFit(PlateAlignment):
    border_padding = None
    padded_img = cropping_rect = None

    status_fitted = False
    status_pad_img = False

    def __init__(self, img, n_rows=8, n_cols=12,
                 blob_detection_method='log',
                 min_sigma=4, max_sigma=40, num_sigma=45,
                 threshold=0.01, overlap=0.1, min_size=200,
                 border_padding=50, verbose=True
                 ):
        self._set_img(img)
        self.border_padding = border_padding
        padded_img = []
        for color_channel in range(3):
            padded_img.append(
                np.expand_dims(
                    np.pad(self.img[:, :, color_channel], self.border_padding, mode='edge' ),
                    axis=2
                )
            )
        self.padded_img = np.concatenate(padded_img, axis=2)
        super().__init__(self.padded_img, n_rows, n_cols,
                         blob_detection_method,
                         min_sigma, max_sigma, num_sigma,
                         threshold, overlap, min_size, verbose)

        self.fit_plate()

    def fit_plate(self):
        if self.status_alignment is False:
            self.align()
        self.verb.start("plate fitting")
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
        self._update_blobs()
        self.status_fitted = True
        self.verb.end("plate fitting")

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

    def plotAx_fitting(self, ax: plt.Axes):
        if self.status_alignment is False:
            self.align()
        if self.status_fitted is False:
            self.fit_plate()

        with plt.ioff():
            ax.grid(False)
            ax.imshow(self.padded_img)
            ax.add_patch(self.cropping_rect)
            ax.set_title("Cropping Outline")
        return ax


    def get_fitted_blob_plot(self):
        with plt.ioff():
            if self.status_fitted is False:
                self.fit_plate()

            fig, ax = plot_blobs(self.img, self.blobs.table, grayscale=False)
        return fig, ax
