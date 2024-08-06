# ----- Imports -----
import pandas as pd

import skimage as ski
from skimage.feature import blob_dog, blob_log, blob_doh
import os
import logging
logger_name = "phenomics-normalization"
log = logging.getLogger(logger_name)
logging.basicConfig(format=f'[%(asctime)s|%(levelname)s|{os.path.basename(__file__)}] %(message)s')
# ----- Pkg Relative Import -----

# ------ Main Class Definition -----
class BlobFinderBase:
    '''
    Last Updated: 7/8/2024

    Note: Only works on float values
    '''


    def __init__(self, gray_img, blob_search_method="log",
                 min_sigma=2, max_sigma=40, num_sigma=30,
                 threshold=0.01, max_overlap=0.1
                 ):
        self._table = None

        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigma = num_sigma

        self.threshold = threshold
        self.max_overlap = max_overlap

        gray_img = self.check_grayscale(gray_img)
        if blob_search_method == "log":
            self._search_blobs_LoG(gray_img)
        elif blob_search_method == "dog":
            self._search_blobs_DoG(gray_img)
        elif blob_search_method == "doh":
            self._search_blobs_DoH(gray_img)
        else:
            self._search_blobs_LoG(gray_img)

    @property
    def table(self):
        return self._table.copy()

    @staticmethod
    def check_grayscale(img):
        if len(img.shape) > 2:
            return ski.color.rgb2gray(img)
        else:
            return img

    def _search_blobs_LoG(self, gray_img):
        log.debug("Starting blob search using 'Laplacian of Gaussian'")

        self._table = pd.DataFrame(blob_log(
            gray_img,
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            num_sigma=self.num_sigma,
            threshold=self.threshold,
            overlap=self.max_overlap
        ), columns=['y', 'x', 'sigma'])

    def _search_blobs_DoG(self, gray_img):
        self._table = pd.DataFrame(
            blob_dog(
                image=gray_img,
                min_sigma=self.min_sigma,
                max_sigma=self.max_sigma,
                threshold=self.threshold,
                overlap=self.max_overlap
            ), columns=["y", 'x', 'sigma']
        )

    def _search_blobs_DoH(self, gray_img):
        self._table = pd.DataFrame(blob_doh(
            gray_img,
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            num_sigma=self.num_sigma,
            threshold=self.threshold,
            overlap=self.max_overlap
        ), columns=['y', 'x', 'sigma'])