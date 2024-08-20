# ----- Imports -----
import pandas as pd
import math

# ----- Pkg Relative Imports -----
from ._blob_finder_base import BlobFinderBase


# ------ Main Class Definition -----
class BlobFinderTable(BlobFinderBase):
    def __init__(self, n_rows=8, n_cols=12,
                 blob_search_method="log",
                 min_sigma=2, max_sigma=35, num_sigma=45,
                 threshold=0.01, max_overlap=0.1
                 ):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.left_bound = self.right_bound = self.upper_bound = self.lower_bound = None

        super().__init__(blob_search_method=blob_search_method,
                         min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma,
                         threshold=threshold, max_overlap=max_overlap)

    @property
    def row_idx(self):
        return range(self.n_rows)

    @property
    def col_idx(self):
        return range(self.n_cols)

    @property
    def rows(self):
        rows = []
        for i in self.row_idx:
            rows.append(
                self.table.loc[
                lambda row: row["row_num"] == i, :
                ]
            )
        return rows

    @property
    def cols(self):
        cols = []
        for i in self.col_idx:
            cols.append(
                self.table.loc[
                lambda row: row["col_num"] == i, :
                ]
            )
        return cols

    def find_blobs(self, img):
        super().find_blobs(img)
        self.generate_table()
        return self.table

    def generate_table(self):
        assert self._table is not None or len(self._table)!=0, "No blobs found in Image"
        self._find_circle_info()
        self._cell_bounds_search()
        self._generate_bins()

    def _find_circle_info(self):
        self._table['radius'] = self._table['sigma'] * math.sqrt(2)
        self._table.drop(columns='sigma')
        self._table['area'] = math.pi * (self._table['radius'] * self._table['radius'])
        if "id" not in self._table.columns:
            self._table = self._table.reset_index(drop=False).rename(columns={
                "index": "id"
            })
        self._table = self._table[["id", 'x', 'y', 'sigma', 'radius', 'area']].reset_index(drop=True)

    def _cell_bounds_search(self):
        self._table['x_minus'] = self._table.x - self._table.radius
        self._table['x_plus'] = self._table.x + self._table.radius
        self._table['y_minus'] = self._table.y - self._table.radius
        self._table['y_plus'] = self._table.y + self._table.radius

    def _generate_bins(self):
        self._table.loc[:, 'row_num'] = pd.cut(
            self._table['y'],
            bins=self.n_rows,
            labels=self.row_idx
        )
        self._table.loc[:, 'col_num'] = pd.cut(
            self._table['x'],
            bins=self.n_cols,
            labels=self.col_idx
        )
        self._table['bin_set'] = "row" + self._table["row_num"].astype(str) \
                                + "_" \
                                + "col" + self._table["col_num"].astype(str)
