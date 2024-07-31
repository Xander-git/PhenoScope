import pandas as pd
from typing import List
import numpy as np

from .plate_profile import PlateProfile
from .colony_profile import cp_connection

import logging

log = logging.getLogger(__file__)
logging.basicConfig(format=f'[%(asctime)s|%(levelname)s|%(name)s] %(message)s')

class PlateSeriesBase:

    def __init__(self, imgs: List[np.ndarray], sample_name, day_index=None,
                 n_rows=8, n_cols=12, align=True, fit=True, auto_analyze=True
                 ):
        cp_connection.refresh()
        self.img_set = imgs
        self.sample_name = sample_name
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.align = align
        self.fit = fit

        self.plates=[]
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
            _tmp = self.get_plate_results(idx).transpose()
            _tmp.index.name = "colony_name"
            results.append(_tmp.reset_index(drop=False))

        results = pd.concat(results, axis=0, ignore_index=True)
        results = results.set_index(["colony_name", "sampling_day"])
        return results

    def run_analysis(self):
        for idx, img in enumerate(self.img_set):
            try:
                log.info(f"Starting plate profiling for plate {idx}")
                cp_api_naming = f"day({self.day_index[idx]})_{self.sample_name}"
                plate = PlateProfile(
                    img=img,
                    sample_name=f"{cp_api_naming}",
                    sampling_day=self.day_index[idx],
                    n_rows=self.n_rows, n_cols=self.n_cols,
                    align=self.align, fit=self.fit,
                    auto_analyze=True
                )
                plate.add_sampling_day(self.day_index[idx])
                self.plates.append(plate)
            except:
                log.warning(f"Could not analyze plate {idx}: {self.sample_name}", exc_info=True)
                self.invalid_imgs.append(img)
        self.status_analysis = True

    def get_results(self):
        return self.results.copy()

    def get_plate_results(self, plate_idx):
        def remove_plate_id(label):
            label = label.split("_")[1:]
            label = "_".join(label)
            return label

        tmp = self.plates[plate_idx].get_results()
        tmp.columns = tmp.columns.map(lambda x: remove_plate_id(x))
        return tmp
