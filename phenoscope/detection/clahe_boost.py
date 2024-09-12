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
from skimage.color import rgb2gray
import numpy as np
from skimage.exposure import equalize_adapthist
from skimage.morphology import white_tophat, disk, square
from skimage.util import img_as_ubyte

from ..util import check_grayscale


class ClaheBoost:
    """
    Takes an image as input and returns a grayscale version of the image
    boosted to improve feature detection and segmentation.
    """

    def __init__(
            self, img, footprint_shape="disk",
            footprint_radius=None,
            kernel_size=None
    ):
        self.img = check_grayscale(img)

        log.debug(f"Setting property footprint_radius from parameter footprint_radius: {footprint_radius}")
        if footprint_radius is None:
            self.footprint_radius = int(np.min(self.img.shape) * 0.002)
        else:
            self.footprint_radius = footprint_radius

        log.debug(f"Creating footprint with radius: {self.footprint_radius}")
        if footprint_shape == "square":
            self.footprint = square(width=(self.footprint_radius * 2))
        elif footprint_shape == "disk":
            self.footprint = disk(radius=self.footprint_radius)
        else:
            self.footprint = disk(radius=self.footprint_radius)

        if kernel_size is None:
            self.kernel_size = int(min(self.img.shape) * (1.0 / 15.0))
        else:
            self.kernel_size = kernel_size

        self.boosted_img = None

        self.use_clahe = True
        self.use_white_tophat = True

        self.status_clahe = False
        self.status_white_tophat = False

    def get_boosted_img(self):
        self._clahe()
        self._white_tophat()
        return img_as_ubyte(self.img)

    def _clahe(self):
        if self.status_clahe is False and self.use_clahe is True:
            self.img = equalize_adapthist(image=self.img,
                                          kernel_size=self.kernel_size
                                          )
            self.status_clahe = True

    def _white_tophat(self):
        if self.status_white_tophat is False and self.use_white_tophat is True:
            tophat_result = white_tophat(image=self.img,
                                         footprint=self.footprint)
            self.img = self.img - tophat_result
            self.status_white_tophat = True
