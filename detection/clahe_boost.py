from skimage.color import rgb2gray
from skimage.exposure import equalize_adapthist
from skimage.morphology import white_tophat, disk, square
from skimage.util import img_as_ubyte


class ClaheBoost:
    """
    Takes an image as input and returns a grayscale version of the image
    boosted to improve feature detection and segmentation.
    """
    def __init__(self, img, footprint_shape="disk",
                 footprint_radius=15,
                 kernel_size=96):
        if len(img.shape) > 2: #2D Check
            self.img = rgb2gray(img)
        else:
            self.img = img

        if footprint_shape == "square":
            self.footprint = square(footprint_radius*2)
        elif footprint_shape == "disk":
            self.footprint = disk(footprint_radius)
        self.kernel_size = kernel_size
        self.footprint_radius = footprint_radius
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
        else:
            pass

    def _white_tophat(self):
        if self.status_white_tophat is False and self.use_white_tophat is True:
            tophat_result = white_tophat(image=self.img,
                                         footprint=self.footprint)
            self.img = self.img - tophat_result
            self.status_white_tophat = True
        else:
            pass
