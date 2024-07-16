# ----- Imports -----
import skimage as ski

# ----- Pkg Relative Import -----
from .cellblobs import CellBlobs
from ..util.verbosity import Verbosity

# ----- Main Class Definition -----
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
