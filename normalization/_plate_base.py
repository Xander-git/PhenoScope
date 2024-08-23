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
from ..util.plotting import plot_plate_rows


# ----- Main Class Definition -----
class PlateBase:
    def __init__(self, img, n_rows=8, n_cols=12):
        self.input_img = img
        self.n_rows = n_rows
        self.n_cols = n_cols
        self._set_img(self.input_img)

        self.status_validity = True

        self._invalid_op = "Normalization run without issues"
        self._invalid_op_img = self._invalid_blobs = None

        log.debug("Initialized Class and Set Image")

    @property
    def gray_img(self):
        return ski.color.rgb2gray(self.img)

    def run(self):
        self.normalize()

    def normalize(self):
        pass

    def _set_img(self, img):
        self.img = ski.util.img_as_ubyte(img)

    def _set_op(self, op_name, op_img, op_blobs):
        self._invalid_op = op_name
        self._invalid_op_img = op_img
        self._invalid_blobs = op_blobs

    def _plotAx_failed_normalization(self, ax):
        if self._invalid_blobs is not None:
            plot_plate_rows(self._invalid_op_img, self._invalid_blobs, ax)
            ax.set_title(self._invalid_op)
        else:
            ax.imshow(self._invalid_op_img)
            ax.set_title(self._invalid_op)
