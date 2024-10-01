import numpy as np
import pandas as pd
import warnings

from ._blob_finder_particle_filter import BlobFinderParticleFilter
from ...util.image_analysis import check_grayscale


class BlobFinderBinMSEFilter(BlobFinderParticleFilter):
    _GRIDROW_MSE_LABEL = 'gridrow_mse'
    _GRIDCOL_MSE_LABEL = 'gridcol_mse'
    _GRID_SECTION_MSE_LABEL = 'grid_section_mse'

    def __init__(
            self, n_rows: int = 8, n_cols: int = 12,
            blob_search_method="log",
            filter_threshold_method="triangle",
            bin_filter_method: str = 'grid_section_mse',
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
    def results(self):
        self._calculate_bin_mse()
        if self.bin_filter_method == 'grid_section_mse':
            # keep only the blobs that contribute the least error
            min_mse_bin_idx = self._table.groupby(by=self._GRID_SECTION_LABEL,
                                                  as_index=True)[f"{self.bin_filter_method}"].idxmin()
            return self._table.loc[min_mse_bin_idx, :]
        elif self.bin_filter_method is None:
            return self._table
        else:
            raise AttributeError('bin_filter_method must be either grid_section_mse or none')

    @property
    def mse(self):
        err = self.results.loc[:, self._GRID_SECTION_MSE_LABEL]
        err = np.sum(err ** 2) / len(err)
        return err

    def find_blobs(self, img):
        img = check_grayscale(img)
        super().find_blobs(img)
        self._calculate_bin_mse()

    def _calculate_bin_mse(self):
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
                other_row_members[self._GRID_SECTION_LABEL] != blob[self._GRID_SECTION_LABEL]
                ]
            row_x = np.array(
                    [blob[self._ROW_COORD_LABEL]] + other_row_members[self._ROW_COORD_LABEL].to_list()
            )
            row_y = np.array(
                    [blob[self._COL_COORD_LABEL]] + other_row_members[self._COL_COORD_LABEL].to_list()
            )
            row_mse[blob.name] = self._get_linreg_mse(row_x, row_y)
            row_residuals = self._get_linreg_residuals(row_x, row_y)

            # get other members
            other_col_members = self._table[
                self._table.loc[:, self._GRIDROW_LABEL] == blob[self._GRIDROW_LABEL]
                ]

            # exclude members in the same gridcol
            other_col_members = other_col_members[
                other_col_members[self._GRID_SECTION_LABEL] != blob[self._GRID_SECTION_LABEL]
                ]

            col_x = np.array(
                    [blob[self._ROW_COORD_LABEL]] + other_col_members[self._ROW_COORD_LABEL].to_list()
            )
            col_y = np.array(
                    [blob[self._COL_COORD_LABEL]] + other_col_members[self._COL_COORD_LABEL].to_list()
            )
            col_mse[blob.name] = self._get_linreg_mse(col_y, col_x)
            col_residuals = self._get_linreg_residuals(col_y, col_x)

            residuals = np.concatenate(
                    [row_residuals, col_residuals]
            )
            mse[blob.name] = np.sum(residuals ** 2) / len(residuals)
        self._table[self._GRIDROW_MSE_LABEL] = pd.Series(row_mse, name=self._GRIDROW_MSE_LABEL)
        self._table[self._GRIDCOL_MSE_LABEL] = pd.Series(col_mse, name=self._GRIDCOL_MSE_LABEL)
        self._table[self._GRID_SECTION_MSE_LABEL] = pd.Series(mse, name=self._GRID_SECTION_MSE_LABEL)

    @staticmethod
    def _get_linreg_residuals(x, y):
        """Get the residuals after performing linear regression."""
        # warnings.simplefilter('ignore', RankWarning)
        warnings.simplefilter('ignore', np.RankWarning)
        m, b = np.polyfit(x, y, 1)
        y_pred = m * x + b
        error = y - y_pred
        return error

    @staticmethod
    def _get_linreg_mse(x, y):
        """Get the mean squared error of a pair of data"""
        # warnings.simplefilter('ignore', RankWarning)
        warnings.simplefilter('ignore', np.RankWarning)
        m, b = np.polyfit(x, y, 1)
        y_pred = m * x + b
        error = np.sum((y - y_pred) ** 2) / len(y)
        return error
