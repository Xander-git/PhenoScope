from skimage.color import rgb2gray

from .plate_io import PlateIO
from ..detection.clahe_boost import ClaheBoost


class PlateBoost(PlateIO):

    @property
    def gray_img(self, boosting_method="clahe"):
        if boosting_method == "none":
            return rgb2gray(self.img)
        if boosting_method == "clahe":
            return ClaheBoost(self.img, kernel_size=(self.n_rows*self.n_cols)).get_boosted_img()
