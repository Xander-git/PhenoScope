import numpy as np
from skimage.color import rgb2gray

from ._plate_io import PlateIO
from phenoscope.detection.clahe_boost import ClaheBoost
from phenoscope.detection import BlobFinder


class PlateBoost(PlateIO):
    def __init__(self, img: np.ndarray, n_rows: int = 8, n_cols: int = 12,
                 border_padding: int = 50,
                 use_boost: bool = True,
                 **kwargs
                 ):
        self._use_boost = use_boost
        self.boost_settings = {
            'footprint_radius':kwargs.get('footprint_radius', 8),
            'footprint_shape':kwargs.get('footprint_shape', 'disk'),
            'kernel_size':kwargs.get('kernel_size', 150),
        }
        super().__init__(
                img=img,
                n_rows=n_rows,
                n_cols=n_cols,
                border_padding=border_padding,
                **kwargs
        )

    @property
    def boosted_img(self):
        return ClaheBoost(img=self.img,
                          footprint_shape=self.boost_settings['footprint_shape'],
                          footprint_radius=self.boost_settings['footprint_radius'],
                          kernel_size=self.boost_settings['kernel_size']
                          ).get_boosted_img()

    def _update_blobs(self):
        if self._use_boost:
            img = self.boosted_img
        else:
            img = self.gray_img
        self.blobs.find_blobs(img)
