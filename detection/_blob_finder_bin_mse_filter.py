import numpy as np
import pandas as pd

from ._blob_finder_particle_filter import BlobFinderParticleFilter


class BlobFinderBinMSEFilter(BlobFinderParticleFilter):
    def __init__(self, gray_img, n_rows=8, n_cols=12,
                 bin_filter_method="mse",
                 blob_search_method="log",
                 filter_threshold_method="triangle",
                 **kwargs
                 ):
        super().__init__(
            gray_img=gray_img,
            n_rows=n_rows, n_cols=n_cols,
            blob_search_method=blob_search_method,
            filter_threshold_method=filter_threshold_method,
            min_sigma=kwargs.get("min_sigma", 4),
            max_sigma=kwargs.get("max_sigma", 40),
            num_sigma=kwargs.get("num_sigma", 45),
            threshold=kwargs.get("threshold", 0.01),
            min_size=kwargs.get("min_size", 180),
            tophat_radius=kwargs.get("tophat_radius", 15),
            border_filter=kwargs.get("border_filter", 50)
        )
        self.bin_filter_method = bin_filter_method
        self.__status_initial_mse = False
        self.__status_update_mse = False
        self._calculate_bin_mse()

    @property
    def table(self):
        self._calculate_bin_mse()
        min_mse_bin_idx = self._table.groupby("bin_set", as_index=True)[f"{self.bin_filter_method}"].idxmin()
        return self._table.loc[min_mse_bin_idx, :]

    @property
    def mse(self):
        err = self.table["mse"]
        err = np.sum(err ** 2) / len(err)
        return err

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
                other_row_members = self._table[
                    self._table.loc[:, "row_num"] == blob["row_num"]
                    ]
                other_row_members = other_row_members[
                    other_row_members["bin_set"] != blob["bin_set"]
                    ]
                row_x = np.array(
                    [blob["x"]] + other_row_members["x"].to_list()
                )
                row_y = np.array(
                    [blob["y"]] + other_row_members["y"].to_list()
                )
                row_mse[blob.name] = self._get_linreg_mse(row_x, row_y)
                row_residuals = self._get_linreg_residuals(row_x, row_y)

                other_col_members = self._table[
                    self._table.loc[:, "col_num"] == blob["col_num"]
                    ]
                other_col_members = other_col_members[
                    other_col_members["bin_set"] != blob["bin_set"]
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
        m, b = np.polyfit(x, y, 1)
        y_pred = m * x + b
        error = y - y_pred
        return error

    @staticmethod
    def _get_linreg_mse(x, y):
        m, b = np.polyfit(x, y, 1)
        y_pred = m * x + b
        error = np.sum((y - y_pred) ** 2) / len(y)
        return error
