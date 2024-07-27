# ----- Imports -----
import pandas as pd
import math

# ----- Pkg Relative Imports -----
from .blob_finder_base import BlobFinderBase

# ------ Main Class Definition -----
class BlobFinderTable(BlobFinderBase):
    '''
    Last Updated: 7/8/2024
    '''
    n_rows = n_cols = None

    row_idx = None
    rows = []

    col_idx = None
    cols = []

    left_bound = right_bound = upper_bound = lower_bound = None
    def __init__(self, img, n_rows=8, n_cols=12, blob_detect_method="log",
                 min_sigma=4, max_sigma=40, num_sigma=45,
                 threshold=0.01, overlap=0.1
                 ):
        super().__init__(img, blob_detect_method,
                         min_sigma, max_sigma, num_sigma,
                         threshold, overlap)
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.generate_table()

    def generate_table(self):
        self._find_circle_info()
        self._cell_bounds_search()
        self._generate_bins()

    def _find_circle_info(self):
        self.table['radius'] = self.table['sigma']*math.sqrt(2)
        self.table.drop(columns='sigma')
        self.table['area'] = math.pi*(self.table['radius']*self.table['radius'])
        self.table = self.table[['x', 'y', 'sigma', 'radius', 'area']].reset_index(drop=True)

    def _cell_bounds_search(self):
        self.table['x_minus'] = self.table.x - self.table.radius
        self.table['x_plus'] = self.table.x + self.table.radius
        self.table['y_minus'] = self.table.y - self.table.radius
        self.table['y_plus'] = self.table.y + self.table.radius

    def _generate_bins(self):
        self.row_idx = list(range(self.n_rows))
        self.col_idx = list(range(self.n_cols))

        self.table.loc[:, 'row_num'] = pd.cut(
            self.table['y'],
            bins=self.n_rows,
            labels=self.row_idx
        )
        self.table.loc[:,'col_num'] = pd.cut(
            self.table['x'],
            bins=self.n_cols,
            labels=self.col_idx
        )

        self.rows = []
        for row_i in self.row_idx:
            self.rows.append(
                self.table[
                    self.table['row_num'] == row_i
                    ]
            )
        self.cols = []
        for col_i in self.col_idx:
            self.cols.append(
                self.table[
                    self.table['col_num'] == col_i
                    ]
            )



