import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage as ski
from skimage import io
from skimage.util import img_as_ubyte
import math
from typing import List

from .plotting import *
from .verbosity import *

###################################################
# ColonyBlobs
###################################################

class CellBlobsBase:
    '''
    Last Updated: 7/8/2024
    '''
    table = None

    min_sigma = max_sigma = num_sigma = None
    threshold = overlap =  None

    verb = None
    def __init__(self, img, blob_detect_method="log",
                 min_sigma=4, max_sigma=40, num_sigma=45,
                 threshold=0.01, overlap=0.1, verbosity = True
                 ):
        self.verb = Verbosity(verbosity)
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigma = num_sigma

        self.threshold = threshold;
        self.overlap = overlap

        if blob_detect_method == "log":
            self.search_blobs_LoG(img)
        elif blob_detect_method == "doh":
            self.search_blobs_DoH(img)
        else:
            self.search_blobs_LoG(img)

    def search_blobs_LoG(self, img):
        self.verb.start("blob finding using LoG")
        self.table = pd.DataFrame(ski.feature.blob_log(
            ski.color.rgb2gray(img),
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            num_sigma=self.num_sigma,
            threshold=self.threshold,
            overlap=self.overlap
        ), columns=['y', 'x', 'sigma'])

    def search_blobs_DoH(self, img):
        self.table = pd.DataFrame(ski.feature.blob_doh(
            ski.color.rgb2gray(img),
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            num_sigma=self.num_sigma,
            threshold=self.threshold,
            overlap=self.overlap
        ), columns=['y', 'x', 'sigma'])


class CellBlobsTable(CellBlobsBase):
    '''
    Last Updated: 7/8/2024
    '''
    n_rows = n_cols = None

    row_idx = None
    rows = []

    col_idx = None
    cols = []

    left_bound = right_bound = upper_bound = lower_bound = None
    def __init__(self, img, n_rows=8, n_cols=12, blob_detect_method="log",
                 min_sigma=4, max_sigma=40, num_sigma=45,
                 threshold=0.01, overlap=0.1, verbose = True
                 ):
        super().__init__(img, blob_detect_method,
                         min_sigma, max_sigma, num_sigma,
                         threshold, overlap, verbose)
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.generate_table()

    def _find_circle_info(self):
        self.table['radius'] = self.table['sigma']*math.sqrt(2)
        self.table.drop(columns='sigma')
        self.table['area'] = math.pi*(self.table['radius']*self.table['radius'])
        self.table = self.table[['x', 'y', 'sigma', 'radius', 'area']].reset_index(drop=True)

    def _cell_bounds_search(self):
        self.table['x_minus'] = self.table.x - self.table.radius
        self.table['x_plus'] = self.table.x + self.table.radius
        self.table['y_minus'] = self.table.y - self.table.radius
        self.table['y_plus'] = self.table.y + self.table.radius

    def _generate_bins(self):
        self.row_idx = list(range(self.n_rows))
        self.col_idx = list(range(self.n_cols))

        self.table.loc[:, 'row_num'] = pd.cut(
            self.table['y'],
            bins=self.n_rows,
            labels=self.row_idx
        )
        self.table.loc[:,'col_num'] = pd.cut(
            self.table['x'],
            bins=self.n_cols,
            labels=self.col_idx
        )

        self.rows = []
        for row_i in self.row_idx:
            self.rows.append(
                self.table[
                    self.table['row_num'] == row_i
                    ]
            )
        self.cols = []
        for col_i in self.col_idx:
            self.cols.append(
                self.table[
                    self.table['col_num'] == col_i
                    ]
            )

    def generate_table(self):
        self._find_circle_info()
        self._cell_bounds_search()
        self._generate_bins()


class CellBlobs(CellBlobsTable):
    '''
    Last Updated: 7/8/2024
    '''

    unfiltered_table = None

    min_size = None
    def __init__(self, img, n_rows=8, n_cols=12, blob_detect_method="log",
                 min_sigma=4, max_sigma=40, num_sigma=45,
                 threshold=0.01, overlap=0.1,
                 min_size=180, verbose = True
                 ):
        super().__init__(img, n_rows, n_cols, blob_detect_method,
                         min_sigma, max_sigma, num_sigma,
                         threshold, overlap, verbose
                         )
        self.min_size = min_size
        self.unfiltered_table = self.table.copy()
        self._filter_by_size()

    def _filter_by_size(self, min_size = None):
        if min_size is None:
            min_size = self.min_size
        self.table = self.table[self.table['area'] >= min_size]
        self.table = self.table.reset_index(drop=True)
        self._generate_bins()

###################################################
# Preprocessing
###################################################
class PlateBase:
    '''
    Last Updated: 7/9/2024
    '''
    input_img = img = gray_img = None

    n_rows = n_cols = None

    blob_detection_method = None
    min_sigma = max_sigma = num_sigma = None
    threshold = overlap = min_size = None

    blobs = None

    verb = None

    def __init__(self, img, n_rows=8, n_cols=12,
                 blob_detection_method="log",
                 min_sigma=4, max_sigma=40, num_sigma=45,
                 threshold=0.01, overlap=0.1, min_size=200,
                 verbose=True
                 ):
        self._set_verbose(verbose)
        self.input_img = img
        self._set_img(img)
        self.blob_detection_method = blob_detection_method

        self.n_rows = n_rows;
        self.n_cols = n_cols
        self.min_size = min_size

        self.min_sigma = min_sigma;
        self.max_sigma = max_sigma
        self.num_sigma = num_sigma
        self.threshold = threshold;
        self.overlap = overlap

        self.verb.start("initial blob search")
        self.verb.end("initial blob search")
        self._update_blobs()

    def _set_verbose(self, verbose=False):
        if type(verbose) == Verbosity:
            self.verb = verbose
        else:
            self.verb = Verbosity(verbose)

    def _set_img(self, img):
        self.img = img
        self.gray_img = ski.color.rgb2gray(img)

    def _update_blobs(self):
        self.blobs = CellBlobs(
            self.img, self.n_rows, self.n_cols, self.blob_detection_method,
            self.min_sigma, self.max_sigma, self.num_sigma, self.threshold, self.overlap,
            self.min_size, verbose=False)


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

        x0 = min(top_row.x);
        y0 = m*x0 + b
        x1 = max(top_row.x);
        y1 = m*x1 + b
        x_align = x1;
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
        self.degree_of_rotation = (
                math.acos(adj/hyp)*(180.0/math.pi)
        )
        if y1>y_align:
            self.degree_of_rotation = self.degree_of_rotation * -1

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


class PlateFit(PlateAlignment):
    '''
    Last Updated: 7/9/2024
    '''
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
        super().__init__(img, n_rows, n_cols,
                         blob_detection_method,
                         min_sigma, max_sigma, num_sigma,
                         threshold, overlap, min_size, verbose)
        self.border_padding = border_padding
        self.fit_plate()

    def fit_plate(self):
        if self.status_alignment is False:
            self.align()
        self.verb.start("plate fitting")
        img = []
        for color_idx in range(3):
            img.append(np.expand_dims(
                np.pad(self.img[:, :, color_idx], (self.border_padding,), mode='edge', ), axis=2))
        img = np.concatenate(img, axis=2)

        self.padded_img = img.copy()
        self._set_img(img)
        self._update_blobs()
        bound_L = math.floor(self.blobs.cols[0].x_minus.min() - self.border_padding)
        bound_R = math.ceil(self.blobs.cols[-1].x_plus.max() + self.border_padding)
        bound_T = math.floor(self.blobs.rows[0].y_minus.min() - self.border_padding)
        bound_B = math.ceil(self.blobs.rows[-1].y_plus.max() + self.border_padding)
        img = img[bound_T:bound_B, bound_L:bound_R]
        with plt.ioff():
            self.cropping_rect = plt.Rectangle((bound_L, bound_T),
                                               bound_R - bound_L,
                                               bound_B - bound_T,
                                               fill=False, edgecolor='white')
        self._set_img(img)
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
            alignFit_ax[0].grid(False)

            alignFit_ax[0].imshow(self.padded_img)
            alignFit_ax[0].add_patch(self.cropping_rect)
            alignFit_ax[0].set_title("Cropping Outline")

            alignFit_ax[1].imshow(self.img)
            alignFit_ax[1].set_title("Fitted Image")
            for idx, row in self.blobs.table.iterrows():
                c = plt.Circle((row['x'], row['y']), row['radius'], color='green', fill=False)
                alignFit_ax[1].add_patch(c)
            alignFit_ax[1].grid(False)
        return alignFit_fig, alignFit_ax

    def plotAx_alignFit(self, ax: plt.Axes):
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


class PlateFit2(PlateAlignment):
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


class WellIsolation(PlateFit2):
    '''
    Last Updated: 7/9/2024
    '''
    cols_midpoints = rows_midpoints = None
    gridded_fig = gridded_ax = None
    well_imgs = None
    invalid_wells = []

    # Status checks
    status_midpoints = False
    status_wells = False

    def __init__(self, img, n_rows=8, n_cols=12,
                 blob_detection_method='log',
                 min_sigma=4, max_sigma=50, num_sigma=30,
                 threshold=0.01, overlap=0.1, min_size=180,
                 border_padding=50, verbose=True
                 ):
        super().__init__(
            img, n_rows, n_cols,
            blob_detection_method,
            min_sigma, max_sigma, num_sigma,
            threshold, overlap, min_size,
            border_padding, verbose
        )
        self.find_midpoints()
        self.find_wells()

    def find_midpoints(self):
        self.verb.start("finding midpoints")
        if self.status_alignment is False:
            self.align()
        if self.status_fitted is False:
            self.fit_plate()
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
        self.verb.end("finding midpoints")

    def find_wells(self):
        if self.status_midpoints is False:
            self.find_midpoints()
        self.verb.start("isolating wells")
        self.well_imgs = []

        y_start = np.insert(self.rows_midpoints, 0, 0).round().astype(int)
        x_start = np.insert(self.cols_midpoints, 0, 0).round().astype(int)
        y_end = np.append(self.rows_midpoints, self.img.shape[0] - 1).round().astype(int)
        x_end = np.append(self.cols_midpoints, self.img.shape[1] - 1).round().astype(int)

        for row_idx in range(self.n_rows):
            for col_idx in range(self.n_cols):
                self.well_imgs.append(
                    self.img[
                    y_start[row_idx]:y_end[row_idx],
                    x_start[col_idx]:x_end[col_idx]
                    ]
                )
        self.status_wells = True
        self.verb.end("isolating wells")

    def set_invalid_well(self, invalid_well_idxes:List[int]):
        self.invalid_wells = self.invalid_wells + invalid_well_idxes

    def plot_well_grid(self, figsize=(12, 8)):
        if self.status_wells is False:
            self.find_midpoints()
        with plt.ioff():
            fig, ax = plt.subplots(figsize=figsize)
            ax = self.plotAx_well_grid(ax)
        return fig, ax

    def plotAx_well_grid(self, ax:plt.Axes):
        if self.status_wells is False:
            self.find_midpoints()
        with plt.ioff():
            ax.imshow(self.img)
            ax.grid(False)
            for point in self.rows_midpoints:
                ax.axhline(point, linestyle="--")
            for point in self.cols_midpoints:
                ax.axvline(point, linestyle='--')
            ax.set_title("Well Isolation Grid")
        return ax

    def plot_well_imgs(self, figsize=(20, 18)):
        if self.status_wells is False:
            self.find_wells()
        with plt.ioff():
            wells_fig, wells_ax = plt.subplots(nrows=self.n_rows, ncols=self.n_cols, tight_layout=True, figsize=figsize)
            for idx, ax in enumerate(wells_ax.ravel()):
                ax.set_title(f"sample_idx: {idx}")
                ax.imshow(self.well_imgs[idx])
                ax.set_axis_off()
        return (wells_fig, wells_ax)


###################################################
# Full Wrapper
###################################################

class Phenotype(WellIsolation): # The parent class will change to the latest endpoint for the pipeline
    '''
    Last Updated: 7/9/2024
    '''
    def __init__(self, img, n_rows=8, n_cols=12,
                 blob_detection_method='log',
                 min_sigma=4, max_sigma=50, num_sigma=30,
                 threshold=0.01, overlap=0.1, min_size=180,
                 border_padding=50, verbose=True, sample_name=None
                 ):
        if sample_name is not None:
            verb = Verbosity(verbose)
            verb.title(sample_name)
        super().__init__(
            img, n_rows, n_cols,
            blob_detection_method,
            min_sigma, max_sigma, num_sigma,
            threshold, overlap, min_size,
            border_padding, verbose
        )

    def imsave(self, fname_save):
        self.verb.start("Saving Plate Image")
        io.imsave(fname_save, img_as_ubyte(self.img), check_contrast=False, quality=100)
        self.verb.end("Saving Plate Image")

    def save_operations(self, fname_save):
        '''
        Saves operation figures to fname_save. This function changes depending on the amound of operations from the start to the endpoint
        '''
        self.verb.start("saving plate operations")
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 16), tight_layout=True)
        opAx = ax.ravel()
        with plt.ioff():
            self.plotAx_alignment(opAx[0])
            self.plotAx_fitting(opAx[1])
            self.plotAx_well_grid(opAx[2])
            opAx[3].imshow(self.img)
            opAx[3].set_title("Final Image")
            fig.savefig(fname_save)
        plt.close(fig)
        self.verb.end("saving plate operations")

    def save_wells(self, fpath_folder, name_prepend="", filetype=".png"):
        if self.status_wells is False:
            self.find_wells()
        self.verb.start("saving isolated well images")
        for idx, well in enumerate(self.well_imgs):
            self.verb.body(f"saving well: {idx}")
            io.imsave(
                f"{fpath_folder}/{name_prepend}well_{idx}{filetype}",
                img_as_ubyte(well), quality=100, check_contrast=False
            )
        self.verb.end("saving isolated well images")
