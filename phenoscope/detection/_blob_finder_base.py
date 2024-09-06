# ----- Imports -----
import numpy as np
import pandas as pd
from skimage.feature import blob_dog, blob_log, blob_doh
import os
import logging

logger_name = "phenomics-normalization"
log = logging.getLogger(logger_name)
logging.basicConfig(format=f'[%(asctime)s|%(levelname)s|{os.path.basename(__file__)}] %(message)s')

# ----- Pkg Relative Import -----
from ..util.image_analysis import check_grayscale


# ------ Main Class Definition -----
class BlobFinderBase:
    def __init__(self,
                 blob_search_method: str = "log",
                 min_sigma: int = 2, max_sigma: int = 40, num_sigma: int = 30,
                 search_threshold: float = 0.01, max_overlap: float = 0.1
                 ):
        self._table = pd.DataFrame()

        self.blob_search_method = blob_search_method
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigma = num_sigma

        self.search_threshold = search_threshold
        self.max_overlap = max_overlap

    @property
    def table(self):
        if self.empty is True: raise ValueError(
                "No blobs are found. Run on an image or boost the image's contrast"
        )
        return self._table.copy()

    @property
    def empty(self):
        return self._table.empty

    def find_blobs(self, img: np.ndarray):
        img = check_grayscale(img)
        self._search_blobs(gray_img=img, method=self.blob_search_method)

    def _search_blobs(self, gray_img, method):
        if method == "log":
            self._search_blobs_LoG(gray_img)
        elif method == "dog":
            self._search_blobs_DoG(gray_img)
        elif method == "doh":
            self._search_blobs_DoH(gray_img)
        else:
            self._search_blobs_LoG(gray_img)

    def _search_blobs_LoG(self, gray_img: np.ndarray):
        log.debug("Starting blob search using 'Laplacian of Gaussian'")

        self._table = pd.DataFrame(blob_log(
                image=gray_img,
                min_sigma=self.min_sigma,
                max_sigma=self.max_sigma,
                num_sigma=self.num_sigma,
                threshold=self.search_threshold,
                overlap=self.max_overlap
        ), columns=['y', 'x', 'sigma'])
        if self._table is None or len(self._table) == 0:
            raise RuntimeError(
                    "No blobs found in image using 'Laplacian of Gaussian'"
            )

    def _search_blobs_DoG(self, gray_img: np.ndarray):
        self._table = pd.DataFrame(blob_dog(
                image=gray_img,
                min_sigma=self.min_sigma,
                max_sigma=self.max_sigma,
                threshold=self.search_threshold,
                overlap=self.max_overlap
        ), columns=["y", 'x', 'sigma'])
        if self._table is None or len(self._table) == 0:
            raise RuntimeError(
                    "No blobs found in image using 'Difference of Gaussian'"
            )

    def _search_blobs_DoH(self, gray_img):
        self._table = pd.DataFrame(blob_doh(
                gray_img,
                min_sigma=self.min_sigma,
                max_sigma=self.max_sigma,
                num_sigma=self.num_sigma,
                threshold=self.search_threshold,
                overlap=self.max_overlap
        ), columns=['y', 'x', 'sigma'])
        if self._table is None or len(self._table) == 0:
            raise RuntimeError(
                    "No blobs found in image using 'Determinant of Hessian'"
            )
