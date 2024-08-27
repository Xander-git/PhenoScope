import logging

formatter = logging.Formatter(
        fmt=f'[%(asctime)s|%(name)s] %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S'
)
console_handler = logging.StreamHandler()
log = logging.getLogger(__name__)
log.addHandler(console_handler)
console_handler.setFormatter(formatter)

import pandas as pd
import numpy as np

from ._plate_series_base import PlateSeriesBase


# TODO: Add module logging

class PlateSeriesChangeOverTime(PlateSeriesBase):
    def get_change_over_time(self):
        if len(self.plates) > 1:
            deltaMetric = []
            for plate_idx in range(len(self._plates.values()) - 1):
                data_zero = self.get_plate_results(
                        plate_idx=plate_idx,
                        numeric_only=True
                ).copy()
                data_zero.loc[:, "sampling_day"] = data_zero.loc[:, "sampling_day"].astype(int)

                data_one = self.get_plate_results(
                        plate_idx=(plate_idx + 1),
                        numeric_only=True
                ).copy()
                data_one.loc[:, "sampling_day"] = data_one.loc[:, "sampling_day"].astype(int)

                tmp = data_one - data_zero
                tmp.columns = tmp.columns.map(lambda x: f"d({x})/dt")

                reference_plates = ("day("
                                    + data_one.loc[:, "sampling_day"].apply(lambda x: f"{x}")
                                    + ") - day("
                                    + data_zero.loc[:, "sampling_day"].apply(lambda x: f"{x}")
                                    + ")")
                tmp.insert(0, "reference_plates", reference_plates)

                tmp.index.name = "colony_name"
                tmp = tmp.reset_index(drop=False)
                tmp = tmp.set_index(["colony_name", "reference_plates"])
                deltaMetric.append(tmp)
            deltaMetric = pd.concat(deltaMetric, axis=0)
            return deltaMetric.sort_index()

    def get_avg_change_over_time(self):
        if len(self._plates) > 1:
            deltaMetric = self.get_change_over_time()
            avg_change = deltaMetric.groupby("colony_name").mean()
            avg_change.columns = avg_change.columns.map(lambda x: f"avg_{x}")

            return avg_change.sort_index()
