# ----- Imports -----
import pandas as pd
import numpy as np
import math
import skimage as ski
import matplotlib.pyplot as plt

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

    # Process Check
    status_alignment = False
    status_alignmentPlot = False

    # Class Constructor
    def __init__(self, img, n_rows=8, n_cols=12,
                 blob_detection_method="log",
                 min_sigma=4, max_sigma=40, num_sigma=45,
                 threshold=0.01, overlap=0.1, min_size=200,
                 verbose=True
                 ):
        super().__init__(img, n_rows, n_cols, blob_detection_method,
                         min_sigma, max_sigma, num_sigma,
                         threshold, overlap, min_size, verbose)

        self.unaligned_blobs = self.blobs
        self.align()



    def align(self):
        self.verb.start("plate alignment")

        top_row = self.blobs.rows[0]

        m, b = np.polyfit(top_row.x, top_row.y, 1)

        x0 = min(top_row.x)
        y0 = m*x0 + b
        x1 = max(top_row.x)
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
                self.input_img,
                self.degree_of_rotation,
                mode='edge'
            )
        )
        self.verb.body("Updating blobs")
        self._update_blobs()
        self.aligned_blobs = self.blobs
        self.status_alignment = True
        self.verb.end("plate alignment")


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

    def plotAx_alignment(self, ax: plt.Axes):
        if self.status_alignment is False:
            self.align()
        with plt.ioff():
            ax.grid(False)
            ax.imshow(self.input_img)
            ax.plot(self.input_alignment_vector['x'], self.input_alignment_vector['y'], color='red')
            ax.plot(
                self.alignment_vector['x'], self.alignment_vector['y'], color='white', linestyle='--'
            )
            ax.set_title(f'Input Alignment Rotation {self.degree_of_rotation} | Red: Original | Yellow: New')
            for idx, row in self.unaligned_blobs.table.iterrows():
                c = plt.Circle((row['x'], row['y']), row['radius'], color='red', fill=False)
                ax.add_patch(c)
            for idx, row in self.aligned_blobs.table.iterrows():
                c = plt.Circle(
                    (row['x'], row['y']), row['radius'], color='yellow', alpha=0.8, fill=False
                )
                ax.add_patch(c)
        return ax
