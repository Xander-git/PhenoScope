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

from ._plate_series_base import PlateSeriesBase


# TODO: Add module logging

class PlateSeriesGrowth(PlateSeriesBase):
    def get_growth(self):
        """
        Calculates and returns the change over time of the plate series.
        This is done at method call in order to maintain flexibility of
        adding more plates over time.
        :return: pd.DataFrame
        """
        if len(self.plates) > 1:
            deltaMetric = []
            for plate_idx in range(len(self._plates.values()) - 1):
                plate_current = self.get_plate_results(
                        plate_idx=plate_idx,
                        numeric_only=True
                ).copy()
                plate_current.loc[:, "SamplingDay"] = plate_current.loc[:, "SamplingDay"].astype(int)

                plate_next = self.get_plate_results(
                        plate_idx=(plate_idx + 1),
                        numeric_only=True
                ).copy()
                plate_next.loc[:, "SamplingDay"] = plate_next.loc[:, "SamplingDay"].astype(int)

                plate_change_table = plate_next - plate_current
                plate_change_table.columns = plate_change_table.columns.map(lambda x: f"d({x})/dt")

                reference_plates = ("day("
                                    + plate_next.loc[:, "SamplingDay"].apply(lambda x: f"{x}")
                                    + ") - day("
                                    + plate_current.loc[:, "SamplingDay"].apply(lambda x: f"{x}")
                                    + ")")
                plate_change_table.insert(0, "reference_plates", reference_plates)

                plate_change_table.index.name = "colony_name"
                plate_change_table = plate_change_table.reset_index(drop=False)
                plate_change_table = plate_change_table.set_index(["colony_name", "reference_plates"])
                deltaMetric.append(plate_change_table)
            deltaMetric = pd.concat(deltaMetric, axis=0)
            return deltaMetric.sort_index()

    def get_avg_growth(self):
        """
        Calculates and returns the average change over time of the plate series.
        This is done at method call in order to maintain flexibility of
        adding more plates over time.
        :return: pd.DataFrame
        """
        if len(self._plates) > 1:
            deltaMetric = self.get_growth()
            avg_change = deltaMetric.groupby("colony_name").mean()
            avg_change.columns = avg_change.columns.map(lambda x: f"avg_{x}")

            return avg_change.sort_index()
