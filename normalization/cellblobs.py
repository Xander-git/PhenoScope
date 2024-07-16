# ----- Imports -----
# ----- Pkg Relative Import -----
from .cellblobs_table import CellBlobsTable


# ----- Main Class Definition -----
# TODO: Add more filters?
class CellBlobs(CellBlobsTable):
    '''
    Last Updated: 7/8/2024
    '''

    unfiltered_table = None

    min_size = None

    def __init__(self, img, n_rows=8, n_cols=12, blob_detect_method="log",
                 min_sigma=4, max_sigma=40, num_sigma=45,
                 threshold=0.01, overlap=0.1,
                 min_size=180, verbose=True
                 ):
        super().__init__(img, n_rows, n_cols, blob_detect_method,
                         min_sigma, max_sigma, num_sigma,
                         threshold, overlap, verbose
                         )
        self.min_size = min_size
        self.unfiltered_table = self.table.copy()
        self._filter_by_size()

    def _filter_by_size(self, min_size=None):
        if min_size is None:
            min_size = self.min_size
        self.table = self.table[self.table['area'] >= min_size]
        self.table = self.table.reset_index(drop=True)
        self.generate_table()
