from .. import Image
from ..interface import GridFeatureExtractor
from . import BoundaryExtractor

from typing import Optional
import pandas as pd


class GridSectionExtractor(GridFeatureExtractor):
    def __init__(self, n_rows: int = 8, n_cols: int = 12):
        self._n_rows: int = n_rows
        self._n_cols: int = n_cols

    def _operate(self, image: Image) -> pd.DataFrame:
        # Find the centroid and boundaries
        results = BoundaryExtractor().extract(image)

        results.loc[:, 'grid_row_bin'] = pd.cut(
                results.loc[:, 'center_rr'],
                bins=self._n_rows,
                labels=range(self._n_rows)
        )

        results.loc[:, 'grid_col_bin'] = pd.cut(
                results.loc[:, 'center_cc'],
                bins=self._n_cols,
                labels=range(self._n_cols)
        )

        results.loc[:, 'grid_section_bin'] = list(zip(
                results.loc[:, 'grid_row_bin'],
                results.loc[:, 'grid_col_bin']
        ))

        results.loc[:, 'grid_section_bin'] = results.loc[:, 'grid_section_bin'].astype('category')

        return results