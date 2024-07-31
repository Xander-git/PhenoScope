# ----- Imports -----
import pandas as pd

import skimage as ski

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
    table = None

    min_sigma = max_sigma = num_sigma = None
    threshold = overlap = None

    def __init__(self, gray_img, blob_detect_method="log",
                 min_sigma=2, max_sigma=40, num_sigma=30,
                 threshold=0.01, overlap=0.1
                 ):
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigma = num_sigma

        self.threshold = threshold
        self.overlap = overlap

        if blob_detect_method == "log":
            self.search_blobs_LoG(gray_img)
        elif blob_detect_method == "doh":
            self.search_blobs_DoH(gray_img)
        else:
            self.search_blobs_LoG(gray_img)

    def search_blobs_LoG(self, gray_img):
        log.debug("Starting blob search using 'Laplacian of Gaussian'")

        self.table = pd.DataFrame(ski.feature.blob_log(
            gray_img,
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            num_sigma=self.num_sigma,
            threshold=self.threshold,
            overlap=self.overlap
        ), columns=['y', 'x', 'sigma'])

    def search_blobs_DoH(self, gray_img):
        self.table = pd.DataFrame(ski.feature.blob_doh(
            gray_img,
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            num_sigma=self.num_sigma,
            threshold=self.threshold,
            overlap=self.overlap
        ), columns=['y', 'x', 'sigma'])