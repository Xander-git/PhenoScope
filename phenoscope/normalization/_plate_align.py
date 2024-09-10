# ----- Logger ------
import logging

formatter = logging.Formatter(
        fmt=f'[%(asctime)s|%(name)s] %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S'
)
console_handler = logging.StreamHandler()
log = logging.getLogger(__name__)
log.addHandler(console_handler)
console_handler.setFormatter(formatter)
# ----- Imports -----
import pandas as pd
import numpy as np
import math
import skimage as ski
import matplotlib.pyplot as plt

# ----- Pkg Relative Import -----
from ._plate_blobs import PlateBlobs

# ----- Class Constants -----
OPERATION_NAME = "PlateAlign"  # TODO: integrate image op stack


# ----- Main Class Definition -----
class PlateAlignment(PlateBlobs):
    def __init__(
            self, img: np.ndarray,
            n_rows: int = 8,
            n_cols: int = 12,
            **kwargs
            ):
        self._unaligned_blobs = None
        self._aligned_blobs = None
        self.input_alignment_vector = None
        self.alignment_vector = None
        self.degree_of_rotation = None

        # execution check
        self._status_alignment = False

        super().__init__(img, n_rows, n_cols, **kwargs)

    def normalize(self, align=True):
        self._normalize(align)
        self._status_normalization = True

    def _normalize(self, align=True):
        super()._normalize()
        if align and self._status_alignment is False: self.align()

    def align(self):
        if self.blobs.empty: self._update_blobs()
        if self.blobs.empty is True: raise ArithmeticError(
                "No blobs were found in the image. Try increasing the contrast of the image or change preprocessing parameters"
        )

        log.info("Starting plate alignment")
        self._unaligned_blobs = self.blobs

        max_row = self.blobs.results.groupby("gridrow_num", observed=True)["mse"].mean()
        max_row = self.blobs.gridrows[max_row.idxmin()]

        # Sets normalization algorithm to use row with the most blobs found
        # Varied performance across different cases

        m, b = np.polyfit(max_row.x, max_row.y, 1)

        x0 = min(max_row.x)
        y0 = m * x0 + b
        x1 = max(max_row.x)
        y1 = m * x1 + b
        x_align = x1
        y_align = y0
        self.input_alignment_vector = pd.DataFrame({
            'x': np.array([x0, x1]),
            'y': np.array([y0, y1])
        })
        self.alignment_vector = pd.DataFrame({
            'x': np.array([x0, x_align]),
            'y': np.array([y0, y_align])
        })

        hyp = np.linalg.norm(np.array([x1, y1]) - np.array([x0, y0]))
        adj = np.linalg.norm(
                np.array([x_align, y_align]) - np.array([x0, y0])
        )

        # Image Y-axis goes top to bottom
        if y1 < y_align:
            self.degree_of_rotation = math.acos(adj / hyp) * (180.0 / math.pi) * -1.0
        else:
            self.degree_of_rotation = math.acos(adj / hyp) * (180.0 / math.pi)

        self._set_img(
                ski.transform.rotate(
                        image=self.img,
                        angle=self.degree_of_rotation,
                        mode='edge'
                )
        )
        log.info("Updating blobs after alignment")
        self._update_blobs()
        self._aligned_blobs = self.blobs
        self._status_alignment = True

    def plot_alignment(self):
        if self._status_alignment is False:
            self.align()
        with plt.ioff():
            alignFit_fig, alignFit_ax = plt.subplots(nrows=1, ncols=2,
                                                     figsize=(14, 10),
                                                     tight_layout=True)

            self.plotAx_alignment(alignFit_ax[0])
            alignFit_ax[1].grid(False)
            alignFit_ax[1].imshow(self.img)
            alignFit_ax[1].set_title("Aligned Image")
        return alignFit_fig, alignFit_ax

    def plotAx_alignment(self, ax: plt.Axes, fontsize=24):
        if self._status_alignment is False:
            self.align()
        with plt.ioff():
            ax.grid(False)
            ax.imshow(self.input_img)
            ax.plot(self.input_alignment_vector['x'],
                    self.input_alignment_vector['y'],
                    color='red')
            ax.plot(self.alignment_vector['x'],
                    self.alignment_vector['y'],
                    color='white', linestyle='--'
                    )
            ax.set_title(f'Input Alignment Rotation {self.degree_of_rotation:.4f} | Red: Original | Yellow: New',
                         fontsize=fontsize)
            for idx, row in self._unaligned_blobs.results.iterrows():
                c = plt.Circle((row['x'], row['y']), row['radius'], color='red', fill=False)
                ax.add_patch(c)
            for idx, row in self._aligned_blobs.results.iterrows():
                c = plt.Circle(
                        (row['x'], row['y']), row['radius'], color='yellow', alpha=0.8, fill=False
                )
                ax.add_patch(c)
        return ax
