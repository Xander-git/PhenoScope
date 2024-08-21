from skimage.color import rgb2gray

from ._plate_io import PlateIO
from ..detection.clahe_boost import ClaheBoost
from ..detection import BlobFinder


class PlateBoost(PlateIO):

    def __init__(self, img, n_rows=8, n_cols=12,
                 align=True, fit=True, use_boost=True,
                 auto_run=True,
                 **kwargs
                 ):
        self._use_boost = use_boost
        self.boost_footprint_radius = kwargs.get("boost_footprint_radius", 8)
        self.boost_footprint_shape = kwargs.get("boost_footprint_shape", "disk")
        self.boost_kernel_size = kwargs.get("boost_kernel_size", 150)
        super().__init__(
            img, n_rows, n_cols,
            align, fit
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
