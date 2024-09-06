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
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
import numpy as np

# ----- Pkg Relative Import -----
from ..util.plotting import plot_plate_rows


# ----- Main Class Definition -----
class PlateBase:
    def __init__(self, img: np.ndarray,
                 n_rows: int = 8,
                 n_cols: int = 12
                 ):
        self.__input_img = None
        self.n_rows = n_rows
        self.n_cols = n_cols
        self._set_img(img)

        self.status_validity = True

        self._status_normalization = False
        self._invalid_op = "Normalization run without issues"
        self._invalid_op_img = self._invalid_blobs = None

        log.debug("Initialized Class and Set Image")

    @property
    def input_img(self):
        return self.__input_img

    @property
    def gray_img(self):
        return rgb2gray(self.img)

    def run(self):
        self.normalize()

    def normalize(self):
        self._normalize()
        self._status_normalization = True

    def _normalize(self):
        pass

    def _set_img(self, img):
        if self.__input_img is None: self.__input_img = img_as_ubyte(img)
        self.img = img_as_ubyte(img)

    def _set_op(self, op_name, op_img, op_blobs):
        self._invalid_op = op_name
        self._invalid_op_img = op_img
        self._invalid_blobs = op_blobs
