# ----- Imports -----
import pandas as pd
import numpy as np
import math
import skimage as ski
import matplotlib.pyplot as plt

import os
import logging
logger_name = "phenomics-normalization"
log = logging.getLogger(logger_name)
logging.basicConfig(format=f'[%(asctime)s|%(levelname)s|{os.path.basename(__file__)}] %(message)s')

# ----- Pkg Relative Import -----

from .plate_base import PlateBase

# ----- Main Class Definition -----


class PlateAlignment(PlateBase):
    '''
    Last Updated: 7/8/2024
    '''
    unaligned_blobs = aligned_blobs = None

    input_alignment_vector = alignment_vector = None

    degree_of_rotation = None

    # Setting
    run_aligment = True

    # Process Check
    status_alignment = False

    # Class Constructor
    def __init__(self, img, n_rows=8, n_cols=12, align=True):
        super().__init__(img, n_rows, n_cols)
        self.run_aligment = align

    def run(self):
        super().run()
        try:
            if self.run_aligment:
                self.align()
        except:
            log.warning("Could not align image")
            self.status_validity = False
            self._invalid_op = "Invalid: Alignment"
            self._invalid_op_img = self.img
            self._invalid_blobs = self.blobs

    def align(self):
        log.info("Starting plate alignment")
        if self.status_initial_blobs is False:
            self._update_blobs()
            self.status_initial_blobs = True
        self.unaligned_blobs = self.blobs
        max_row = self.blobs.rows[0]

        # Sets normalization algorithm to use row with the most blobs found
        # Varied performance across different cases
        #
        for row in self.blobs.rows[1:]:
            if len(max_row) < len(row):
                max_row = row

        m, b = np.polyfit(max_row.x, max_row.y, 1)

        x0 = min(max_row.x)
        y0 = m*x0 + b
        x1 = max(max_row.x)
        y1 = m*x1 + b
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
            self.degree_of_rotation = math.acos(adj/hyp)*(180.0/math.pi)*-1.0
        else:
            self.degree_of_rotation = math.acos(adj/hyp)*(180.0/math.pi)

        self._set_img(
            ski.transform.rotate(
                self.img,
                self.degree_of_rotation,
                mode='edge'
            )
        )
        log.info("Updating blobs after alignment")
        self._update_blobs()
        self.aligned_blobs = self.blobs
        self.status_alignment = True

    def plot_alignment(self):
        if self.status_alignment is False:
            self.align()
        with plt.ioff():
            alignFit_fig, alignFit_ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 10),
                                                     tight_layout=True)

            self.plotAx_alignment(alignFit_ax[0])
            alignFit_ax[1].grid(False)
            alignFit_ax[1].imshow(self.img)
            alignFit_ax[1].set_title("Aligned Image")
        return alignFit_fig, alignFit_ax

    def plotAx_alignment(self, ax: plt.Axes, fontsize=24):
        if self.status_alignment is False:
            self.align()
        with plt.ioff():
            ax.grid(False)
            ax.imshow(self.input_img)
            ax.plot(self.input_alignment_vector['x'], self.input_alignment_vector['y'], color='red')
            ax.plot(
                self.alignment_vector['x'], self.alignment_vector['y'], color='white', linestyle='--'
            )
            ax.set_title(f'Input Alignment Rotation {self.degree_of_rotation:.4f} | Red: Original | Yellow: New',fontsize=fontsize)
            for idx, row in self.unaligned_blobs.table.iterrows():
                c = plt.Circle((row['x'], row['y']), row['radius'], color='red', fill=False)
                ax.add_patch(c)
            for idx, row in self.aligned_blobs.table.iterrows():
                c = plt.Circle(
                    (row['x'], row['y']), row['radius'], color='yellow', alpha=0.8, fill=False
                )
                ax.add_patch(c)
        return ax
