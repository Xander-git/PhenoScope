# ----- Imports -----
import skimage as ski

import os
import logging
logger_name = "phenomics-normalization"
log = logging.getLogger(logger_name)
logging.basicConfig(format=f'[%(asctime)s|%(levelname)s|{os.path.basename(__file__)}] %(message)s')
# ----- Pkg Relative Import -----
from ..detection.blob_finder import BlobFinder
from ..util.plotting import plot_plate_rows

# ----- Main Class Definition -----
class PlateBase:
    '''
    Last Updated: 7/9/2024
    '''
    blobs_detection_method = "log"
    blobs_min_sigma = 4
    blobs_max_sigma = 40
    blobs_num_sigma = 45
    blobs_threshold = 0.01
    blobs_overlap = 0.1
    blobs_min_size = 180
    blobs_filter_threshold_method = "triangle"
    blobs_tophat_radius = 15
    blobs_border_filter = 50

    blobs = None


    status_initial_blobs = False
    status_validity = True
    
    _invalid_op = "Normalization run without any issues"
    _invalid_op_img = None
    _invalid_blobs = None
    def __init__(self, img, n_rows=8, n_cols=12):
        self.input_img = img
        self.img = ski.util.img_as_ubyte(img)
        self.n_rows = n_rows
        self.n_cols = n_cols
        log.debug("Initialized Class and Set Image")

    @property
    def gray_img(self):
        return ski.color.rgb2gray(self.img)

    def run(self):
        # TODO: Integrate autorun switch
        try:
            log.info("Starting initial blob search")
            self._update_blobs()
            self.status_initial_blobs = True
        except:
            self.status_validity = False
            self._invalid_op = "Initial Blob Search"
            self._invalid_op_img = self.img

    def _set_img(self, img):
        self.img = ski.util.img_as_ubyte(img)

    def _update_blobs(self):
        self.blobs = BlobFinder(
            gray_img=self.gray_img,
            n_rows=self.n_rows,
            n_cols=self.n_cols,
            blob_search_method=self.blobs_detection_method,
            min_sigma=self.blobs_min_sigma,
            max_sigma=self.blobs_max_sigma,
            num_sigma=self.blobs_num_sigma,
            threshold=self.blobs_threshold,
            max_overlap=self.blobs_overlap,
            min_size=self.blobs_min_size,
            filter_threshold_method= self.blobs_filter_threshold_method,
            tophat_radius=self.blobs_tophat_radius,
            border_filter=self.blobs_border_filter
        )

    def _plotAx_failed_normalization(self, ax):
        if self._invalid_blobs is not None:
            plot_plate_rows(self._invalid_op_img, self._invalid_blobs, ax)
            ax.set_title(self._invalid_op)
        else:
            ax.imshow(self._invalid_op_img)
            ax.set_title(self._invalid_op)

