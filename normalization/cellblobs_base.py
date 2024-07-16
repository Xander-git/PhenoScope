# ----- Imports -----
import pandas as pd

import skimage as ski
# ----- Pkg Relative Import -----
from ..util.verbosity import Verbosity

# ------ Main Class Definition -----
class CellBlobsBase:
    '''
    Last Updated: 7/8/2024
    '''
    table = None

    min_sigma = max_sigma = num_sigma = None
    threshold = overlap =  None

    verb = None
    def __init__(self, img, blob_detect_method="log",
                 min_sigma=4, max_sigma=40, num_sigma=45,
                 threshold=0.01, overlap=0.1, verbosity = True
                 ):
        self.verb = Verbosity(verbosity)
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigma = num_sigma

        self.threshold = threshold;
        self.overlap = overlap

        if blob_detect_method == "log":
            self.search_blobs_LoG(img)
        elif blob_detect_method == "doh":
            self.search_blobs_DoH(img)
        else:
            self.search_blobs_LoG(img)

    def search_blobs_LoG(self, img):
        self.verb.start("blob finding using LoG")
        self.table = pd.DataFrame(ski.feature.blob_log(
            ski.color.rgb2gray(img),
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            num_sigma=self.num_sigma,
            threshold=self.threshold,
            overlap=self.overlap
        ), columns=['y', 'x', 'sigma'])

    def search_blobs_DoH(self, img):
        self.table = pd.DataFrame(ski.feature.blob_doh(
            ski.color.rgb2gray(img),
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            num_sigma=self.num_sigma,
            threshold=self.threshold,
            overlap=self.overlap
        ), columns=['y', 'x', 'sigma'])