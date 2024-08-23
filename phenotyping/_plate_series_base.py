import pandas as pd
from typing import List
import numpy as np

from .plate_profile import PlateProfile

import logging

log = logging.getLogger(__file__)
logging.basicConfig(format=f'[%(asctime)s|%(levelname)s|%(name)s] %(message)s')


class PlateSeriesBase:

    def __init__(self, imgs: List[np.ndarray], sample_name: List[str], day_index: List[int] = None,
                 n_rows=8, n_cols=12, align=True, fit=True, auto_analyze=True
                 ):
        self.img_set = imgs
        self.sample_name = sample_name
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.align = align
        self.fit = fit

        self.plates = []
        self.invalid_imgs = []

        self.status_analysis = False

        if day_index is None:
            self.day_index = range(1, len(imgs) + 1)
        else:
            self.day_index = day_index

        if auto_analyze: self.run_analysis()

    @property
    def results(self):
        if self.status_analysis is False: self.run_analysis()

        results = []
        for idx in range(len(self.plates)):
            _tmp = self.get_plate_results(idx)
            _tmp.index.name = "colony_name"
            results.append(_tmp.reset_index(drop=False))

        results = pd.concat(results, axis=0, ignore_index=True)
        results = results.set_index(["colony_name", "sampling_day"])
        return results

    def run_analysis(self):
        for idx, img in enumerate(self.img_set):
            try:
                log.info(f"Starting plate profiling for plate {idx}")
                cp_api_naming = self._add_day_to_name(day=self.day_index[idx], name=self.sample_name)
                plate = PlateProfile(
                        img=img,
                        plate_name=f"{cp_api_naming}",
                        sampling_day=self.day_index[idx],
                        n_rows=self.n_rows, n_cols=self.n_cols,
                        align=self.align, fit=self.fit,
                        auto_analyze=True
                )
                self.plates.append(plate)
            except:
                log.warning(f"Could not analyze plate {idx}: {self.sample_name}", exc_info=True)
                self.invalid_imgs.append(img)
        self.status_analysis = True

    def get_results(self):
        return self.results

    def get_plate_results(self, plate_idx, numeric_only=False, include_adv=False):
        tmp = self.plates[plate_idx].get_results(
                numeric_only=numeric_only,
                include_adv=include_adv
        )

        tmp.index = tmp.index.map(lambda x: self._remove_day_from_name(x))
        return tmp

    @staticmethod
    def _add_day_to_name(day, name):
        return f"day({day})_{name}"

    @staticmethod
    def _remove_day_from_name(dayname):
        name = dayname.split("_")[1:]
        name = "_".join(name)
        return name
