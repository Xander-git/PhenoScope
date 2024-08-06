import pandas as pd
import numpy as np

from .plate_series_base import PlateSeriesBase

# TODO: Add module logging

class PlateSeriesChangeOverTime(PlateSeriesBase):
    def get_change_over_time(self):
        if len(self.plates) > 1:
            deltaMetric = []
            for idx in range(len(self.plates) - 1):
                data_zero = self.get_plate_results(idx).drop(
                    index=["status_valid_analysis"]
                ).select_dtypes(include=np.number)
                data_one = self.get_plate_results(idx + 1).drop(
                    index=["status_valid_analysis"]
                ).select_dtypes(include=np.number)
                tmp = data_one - data_zero
                tmp.index = tmp.index.map(lambda x: f"d({x})/dt")
                reference_plates = np.full(len(tmp.index), f"plate{idx + 1})-plate({idx})")
                tmp.insert(0, "reference_plates", reference_plates)
                tmp = tmp.reset_index(drop=False)
                tmp = tmp.rename(columns={
                    "index": "d(metric)/dt"
                })
                tmp = tmp.set_index(["reference_plates", "d(metric)/dt"])
                deltaMetric.append(tmp)
            deltaMetric = pd.concat(deltaMetric, axis=0)
            return deltaMetric.sort_index()

    def get_avg_change_over_time(self):
        if len(self.plates) > 1:
            deltaMetric = self.get_change_over_time()
            avg_change = deltaMetric.groupby("d(metric)/dt").mean()
            return avg_change.sort_index()
