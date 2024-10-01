from typing import List

import numpy as np
import pandas as pd

from ._threshold_finder_measurement_filter import ThresholdFinderMeasurementFilter

GRIDROW_LABEL = "gridrow_num"
GRIDCOL_LABEL = "gridcol_num"
BIN_SET_LABEL = "bin_set"


class ThresholdFinderBins(ThresholdFinderMeasurementFilter):
    def __init__(
            self,
            threshold_method: str = "otsu",
            n_rows: int = 8,
            n_cols: int = 12,
            enhance_contrast: bool = True,
            **kwargs
            ):
        super().__init__(
                threshold_method=threshold_method,
                enhance_contrast=enhance_contrast,
                **kwargs
        )
        self._n_rows: int = n_rows
        self._n_cols: int = n_cols

    @property
    def n_rows(self) -> int:
        return self._n_rows

    @n_rows.setter
    def n_rows(self, value: int):
        self._n_rows = value
        if BIN_SET_LABEL in self._table.columns:
            self.generate_bins()

    @property
    def n_cols(self) -> int:
        return self._n_cols

    @n_cols.setter
    def n_cols(self, value: int):
        self._n_cols = value
        if BIN_SET_LABEL in self._table.columns:
            self.generate_bins()

    @property
    def bin_values(self) -> np.ndarray:
        bin_values = np.array(self._table.loc[:, BIN_SET_LABEL].unique())
        return np.sort(bin_values)

    @property
    def bins(self) -> List[pd.DataFrame]:
        table = self.results
        return [
            table.loc[
            table.loc[:, BIN_SET_LABEL] == grid_bin, :]
            for grid_bin in self.bin_values
        ]

    @property
    def gridrows(self) -> List[pd.DataFrame]:
        table = self.results
        return [
            table.loc[
            table.loc[:, GRIDROW_LABEL] == i, :]
            for i in range(self.n_rows)
        ]

    @property
    def gridcols(self) -> List[pd.DataFrame]:
        table = self.results
        return [
            table.loc[
            table.loc[:, GRIDCOL_LABEL] == i, :]
            for i in range(self.n_cols)
        ]

    def find_objects(self, image: np.ndarray) -> None:
        super().find_objects(image=image)
        self.generate_bins()

    def generate_bins(self) -> None:
        error_msg = \
            """
            Failed to generate bins due to empty table. 
            Try running find_objects() on an image first.
            """

        if self._table.empty: raise AttributeError(error_msg)

        self._table.loc[:, GRIDROW_LABEL] = pd.cut(
                x=self._table.loc[:, "center-rr"],
                bins=self.n_rows, labels=range(self.n_rows)
        )

        self._table.loc[:, GRIDCOL_LABEL] = pd.cut(
                x=self._table.loc[:, "center-cc"],
                bins=self.n_cols, labels=range(self.n_cols)
        )

        self._table.loc[:, BIN_SET_LABEL] = \
            "gridrow" \
            + self._table.loc[:, GRIDROW_LABEL].astype(str).str.zfill(3) \
            + "_" + "gridcol" \
            + self._table.loc[:, GRIDCOL_LABEL].astype(str).str.zfill(3)