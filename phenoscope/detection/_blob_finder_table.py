# ----- Imports -----
import pandas as pd
import math

# ----- Pkg Relative Imports -----
from ._blob_finder_base import BlobFinderBase


# ------ Main Class Definition -----
class BlobFinderTable(BlobFinderBase):
    _GRIDROW_LABEL = "gridrow_num"
    _GRIDCOL_LABEL = "gridcol_num"
    _BIN_SET_LABEL = "bin_set"

    def __init__(self, n_rows: int = 8, n_cols: int = 12,
                 blob_search_method: str = "log",
                 min_sigma: int = 2, max_sigma: int = 35, num_sigma: int = 45,
                 search_threshold: float = 0.01, max_overlap: float = 0.1
                 ):
        self.n_rows = n_rows
        self.n_cols = n_cols

        super().__init__(blob_search_method=blob_search_method,
                         min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma,
                         search_threshold=search_threshold, max_overlap=max_overlap)

    @property
    def gridrows(self):
        return [
            self.results.loc[
            self.results.loc[:, self._GRIDROW_LABEL] == i, :]
            for i in range(self.n_rows)
        ]

    @property
    def gridcols(self):
        return [
            self.results.loc[
            self.results.loc[:, self._GRIDCOL_LABEL] == i, :]
            for i in range(self.n_cols)
        ]

    def find_blobs(self, img):
        super().find_blobs(img)
        self.generate_table()
        return self.results

    def generate_table(self):
        if self.empty is True: raise ValueError(
                "No blobs are found. Run on an image or boost the image's contrast"
        )

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
        self._table.loc[:, self._GRIDROW_LABEL] = pd.cut(
                self._table['y'],
                bins=self.n_rows,
                labels=range(self.n_rows)
        )
        self._table.loc[:, self._GRIDCOL_LABEL] = pd.cut(
                self._table['x'],
                bins=self.n_cols,
                labels=range(self.n_cols)
        )
        self._table[self._BIN_SET_LABEL] = "gridrow" + self._table[self._GRIDROW_LABEL].astype(str) \
                                           + "_" \
                                           + "gridcol" + self._table[self._GRIDCOL_LABEL].astype(str)
