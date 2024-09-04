from typing import List

import numpy as np
import pandas as pd

from ._object_finder_measurement_filter import ObjectFinderMeasurementFilter


class ObjectFinderBins(ObjectFinderMeasurementFilter):
    def __init__(self, img: np.ndarray, n_rows: int, n_cols: int,
                 boost_segmentation: bool = True,
                 **kwargs
                 ):
        super().__init__(img, boost_segmentation, **kwargs)
        self._n_rows = n_rows
        self._n_cols = n_cols

    @property
    def n_rows(self):
        return self._n_rows

    @n_rows.setter
    def n_rows(self, value: int):
        self._n_rows = value
        if "bin_set" in self._table.columns:
            self.generate_bins()

    @property
    def n_cols(self):
        return self._n_cols

    @n_cols.setter
    def n_cols(self, value: int):
        self._n_cols = value
        if "bin_set" in self._table.columns:
            self.generate_bins()

    @property
    def bin_values(self):
        bin_values = np.array(self._table.loc[:, "bin_set"].unique())
        return np.sort(bin_values)

    @property
    def bins(self):
        table = self.results
        return [
            table.loc[
            table.loc[:, "bin_set"] == grid_bin, :]
            for grid_bin in self.bin_values
        ]

    @property
    def rows(self):
        table = self.results
        return [
            table.loc[
            table.loc[:, "row_num"] == i, :]
            for i in range(self.n_rows)
        ]

    @property
    def cols(self):
        return [
            self._table.loc[
            self._table.loc[:, "col_num"] == i, :]
            for i in range(self.n_cols)
        ]

    def find_objects(self, threshold_method: str = "otsu",
                     measurements: str or List[str] = "basic",
                     **kwargs
                     ):
        super().find_objects(threshold_method=threshold_method,
                             measurements=measurements,
                             **kwargs)
        self.generate_bins()

    def generate_bins(self):
        self._table.loc[:, "row_num"] = pd.cut(
                self._table.loc[:, "centroid-row"],
                bins=self.n_rows, labels=range(self.n_rows)
        )

        self._table.loc[:, "col_num"] = pd.cut(
                self._table.loc[:, "centroid-col"],
                bins=self.n_cols, labels=range(self.n_cols)
        )

        self._table.loc[:, "bin_set"] = \
            "row" \
            + self._table.loc[:, "row_num"].astype(str).str.zfill(3) \
            + "_" + "col" \
            + self._table.loc[:, "col_num"].astype(str).str.zfill(3)
