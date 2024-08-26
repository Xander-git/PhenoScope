import pandas as pd
from typing import List
import numpy as np

from .plate_profile import PlateProfile

import logging

log = logging.getLogger(__file__)
logging.basicConfig(format=f'[%(asctime)s|%(levelname)s|%(name)s] %(message)s')


class PlateSeriesBase:

    def __init__(self, series_name: str, n_rows=8, n_cols=12,
                 ):
        self.series_name = series_name
        self.n_rows = n_rows
        self.n_cols = n_cols

        self._plates = {}

    @property
    def results(self):
        # TODO: Finish how results are collected
        if self.status_analysis is False: self.run_analysis()

        results = []
        for idx in range(len(self._plates)):
            _tmp = self.get_plate_results(idx)
            _tmp.index.name = "colony_name"
            results.append(_tmp.reset_index(drop=False))

        results = pd.concat(results, axis=0, ignore_index=True)
        results.insert(0, "series_name", value=self.series_name)
        results = results.set_index(["colony_name", "sampling_day"])
        return results

    @property
    def status_analysis(self):
        status = True
        for plate in self.plates:
            if plate.status_measure is False: status = False
        return status

    @property
    def plate_idx(self):
        idx = [self._plates.keys()]
        idx.sort()
        return idx

    @property
    def plates(self):
        return [self.plates[key] for key in self.plate_idx]

    def add_plate(self, plate_profile: PlateProfile):
        assert plate_profile._sampling_day != np.nan, "PlateProfile is missing a sampling day"
        self._plates[int(plate_profile._sampling_day)] = plate_profile

    def add_plate_img(self, img: np.ndarray, plate_name: str, sampling_day: int,
                      align: bool = True,
                      fit: bool = True,
                      measure=True,
                      use_boost=True,
                      **kwargs
                      ):
        plate = PlateProfile(
                img=img,
                plate_name=plate_name,
                sampling_day=sampling_day,
                n_rows=self.n_rows,
                n_cols=self.n_cols,
                use_boose=use_boost,
                **kwargs
        )
        if align or fit: plate.normalize(align=align, fit=fit)
        if measure: plate.run(align=align, fit=fit)
        self.add_plate(plate)

    def run_analysis(self):
        for plate in self.plates:
            plate.run()

    def get_results(self):
        return self.results

    def get_plate_results(self, plate_idx, numeric_only=False, include_adv=False):
        tmp = self._plates[plate_idx].get_results(
                numeric_only=numeric_only,
                include_adv=include_adv
        )
        return tmp
