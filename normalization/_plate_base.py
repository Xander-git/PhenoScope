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
import skimage as ski

# ----- Pkg Relative Import -----
from ..detection.blob_finder import BlobFinder
from ..util.plotting import plot_plate_rows


# ----- Main Class Definition -----
class PlateBase:
    def __init__(self, img, n_rows=8, n_cols=12, **kwargs):
        self.input_img = img
        self.n_rows = n_rows
        self.n_cols = n_cols
        self._set_img(self.input_img)

        self.blobs_detection_method = kwargs.get("blobs_detection_method", "log")
        self.blobs_min_sigma = kwargs.get("blobs_min_sigma", 4)
        self.blobs_max_sigma = kwargs.get("blobs_max_sigma", 40)
        self.blobs_num_sigma = kwargs.get("blobs_num_sigma", 45)
        self.blobs_threshold = kwargs.get("blobs_threshold", 0.01)
        self.blobs_overlap = kwargs.get("blobs_overlap", 0.1)
        self.blobs_min_size = kwargs.get("blobs_min_size", 180)
        self.blobs_filter_threshold_method = kwargs.get("blobs_filter_threshold_method", "triangle")
        self.blobs_tophat_radius = kwargs.get("blobs_tophat_radius", 15)
        self.blobs_border_filter = kwargs.get("blobs_border_filter", 50)

        self.blobs = None

        self.status_initial_blobs = False
        self.status_validity = True

        self._invalid_op = "Normalization run without issues"
        self._invalid_op_img = self._invalid_blobs = None

        log.debug("Initialized Class and Set Image")

    @property
    def gray_img(self):
        return ski.color.rgb2gray(self.img)

    def run(self):
        # TODO: Integrate autorun
        log.info("Starting initial blob search")
        self._update_blobs()
        self.status_initial_blobs = True

        # try:
        #     log.info("Starting initial blob search")
        #     self._update_blobs()
        #     self.status_initial_blobs = True
        # except:
        #     self.status_validity = False
        #     self._invalid_op = "Initial Blob Search"
        #     self._invalid_op_img = self.img

    def _set_img(self, img):
        self.img = ski.util.img_as_ubyte(img)

    def _set_op(self, op_name, op_img, op_blobs):
        self._invalid_op = op_name
        self._invalid_op_img = op_img
        self._invalid_blobs = op_blobs

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
                filter_threshold_method=self.blobs_filter_threshold_method,
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
