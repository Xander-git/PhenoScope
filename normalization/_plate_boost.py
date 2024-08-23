import numpy as np
from skimage.color import rgb2gray

from ._plate_io import PlateIO
from ..detection.clahe_boost import ClaheBoost
from ..detection import BlobFinder


class PlateBoost(PlateIO):

    def __init__(self, img: np.ndarray, n_rows: int = 8, n_cols: int = 12,
                 border_padding: int = 50,
                 use_boost: bool = True,
                 auto_run=True,
                 **kwargs
                 ):
        self._use_boost = use_boost
        self.boost_footprint_radius = kwargs.get("boost_footprint_radius", 8)
        self.boost_footprint_shape = kwargs.get("boost_footprint_shape", "disk")
        self.boost_kernel_size = kwargs.get("boost_kernel_size", 150)
        super().__init__(
                img=img,
                n_rows=n_rows,
                n_cols=n_cols,
                border_padding=border_padding,
                auto_run=auto_run,
                **kwargs
        )
        if auto_run:
            self.run()

    @property
    def boosted_img(self):
        return ClaheBoost(img=self.img,
                          footprint_shape=self.boost_footprint_shape,
                          footprint_radius=self.boost_footprint_radius,
                          kernel_size=self.boost_kernel_size
                          ).get_boosted_img()

    def _update_blobs(self):
        if self._use_boost:
            img = self.boosted_img
        else:
            img = self.gray_img
        self.blobs.find_blobs(img)
