import numpy as np
import pandas as pd
import warnings

from numpy.exceptions import RankWarning

from ._blob_finder_particle_filter import BlobFinderParticleFilter
from ..util.image_analysis import check_grayscale


class BlobFinderBinMSEFilter(BlobFinderParticleFilter):
    def __init__(self, n_rows: int = 8, n_cols: int = 12,
                 blob_search_method="log",
                 filter_threshold_method="triangle",
                 bin_filter_method: str = "mse",
                 tophat_shape="square",
                 tophat_radius=12,
                 min_area=180,
                 **kwargs
                 ):
        super().__init__(
                n_rows=n_rows, n_cols=n_cols,
                min_area=min_area,
                filter_threshold_method=filter_threshold_method,
                tophat_shape=tophat_shape,
                tophat_radius=tophat_radius,
                border_filter=kwargs.get("border_filter", 50),
                blob_search_method=blob_search_method,
                min_sigma=kwargs.get("min_sigma", 4),
                max_sigma=kwargs.get("max_sigma", 40),
                num_sigma=kwargs.get("num_sigma", 45),
                search_threshold=kwargs.get("search_threshold", 0.01),

        )
        self.bin_filter_method = bin_filter_method
        self.__status_initial_mse = False
        self.__status_update_mse = False

    @property
    def table(self):
        self._calculate_bin_mse()
        min_mse_bin_idx = self._table.groupby(by=self._BIN_SET_LABEL, as_index=True)[f"{self.bin_filter_method}"].idxmin()
        return self._table.loc[min_mse_bin_idx, :]

    @property
    def mse(self):
        err = self.table.loc[:, "mse"]
        err = np.sum(err ** 2) / len(err)
        return err

    def find_blobs(self, img):
        self.__status_initial_mse = False
        self.__status_update_mse = False
        img = check_grayscale(img)
        super().find_blobs(img)
        self._calculate_bin_mse()

    def _calculate_bin_mse(self):
        if (
                self.__status_initial_mse is False
                or
                self.__status_update_mse is False
        ):

            row_mse = {}
            col_mse = {}
            mse = {}
            for idx in range(self._table.shape[0]):
                blob = self._table.iloc[idx, :]

                # get other members
                other_row_members = self._table[
                    self._table.loc[:, self._GRIDROW_LABEL] == blob[self._GRIDCOL_LABEL]
                    ]

                # exclude other members in the same gridrow
                other_row_members = other_row_members[
                    other_row_members[self._BIN_SET_LABEL] != blob[self._BIN_SET_LABEL]
                    ]
                row_x = np.array(
                        [blob["x"]] + other_row_members["x"].to_list()
                )
                row_y = np.array(
                        [blob["y"]] + other_row_members["y"].to_list()
                )
                row_mse[blob.name] = self._get_linreg_mse(row_x, row_y)
                row_residuals = self._get_linreg_residuals(row_x, row_y)

                # get other members
                other_col_members = self._table[
                    self._table.loc[:, self._GRIDROW_LABEL] == blob[self._GRIDROW_LABEL]
                    ]

                # exclude members in the same gridcol
                other_col_members = other_col_members[
                    other_col_members[self._BIN_SET_LABEL] != blob[self._BIN_SET_LABEL]
                    ]

                col_x = np.array(
                        [blob["x"]] + other_col_members["x"].to_list()
                )
                col_y = np.array(
                        [blob["y"]] + other_col_members["y"].to_list()
                )
                col_mse[blob.name] = self._get_linreg_mse(col_y, col_x)
                col_residuals = self._get_linreg_residuals(col_y, col_x)

                residuals = np.concatenate(
                        [row_residuals, col_residuals]
                )
                mse[blob.name] = np.sum(residuals ** 2) / len(residuals)
            row_mse = pd.Series(row_mse, name="row_mse")
            col_mse = pd.Series(col_mse, name="col_mse")
            mse = pd.Series(mse, name="mse")
            if (
                    all(
                            label in self._table.columns
                            for label in
                            ["row_mse", "col_mse", "mse"]
                    )
                    and
                    self.__status_initial_mse is True
            ):
                self._table.loc[:, "row_mse"] = row_mse
                self._table.loc[:, "col_mse"] = col_mse
                self._table.loc[:, "mse"] = mse
                self.__status_update_mse = True
            else:
                self._table = pd.concat(
                        [self._table, row_mse, col_mse, mse],
                        axis=1
                )
                self.__status_initial_mse = True

    @staticmethod
    def _get_linreg_residuals(x, y):
        warnings.simplefilter('ignore', RankWarning)
        m, b = np.polyfit(x, y, 1)
        y_pred = m * x + b
        error = y - y_pred
        return error

    @staticmethod
    def _get_linreg_mse(x, y):
        warnings.simplefilter('ignore', RankWarning)
        m, b = np.polyfit(x, y, 1)
        y_pred = m * x + b
        error = np.sum((y - y_pred) ** 2) / len(y)
        return error
