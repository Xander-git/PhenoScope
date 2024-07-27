import pandas as pd
from typing import List
import numpy as np

from .plate_profile import PlateProfile

import logging

log = logging.getLogger("phenomics-phenotyping")


class PlateSeriesBase:
    img_set = None
    sample_name = None
    day_index = None
    n_rows = n_cols = None
    align = fit = None

    plates = []
    results = None

    status_analyze = False
    invalid_imgs = []

    def __init__(self, imgs: List[np.ndarray], sample_name, day_index=None, n_rows=8, n_cols=12, align=True, fit=True, auto_analyze=True):
        self.img_set = imgs
        self.sample_name = sample_name
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.align = align
        self.fit = fit
        if day_index is None:
            self.day_index = range(1, len(imgs) + 1)
        else:
            self.day_index = day_index

        if auto_analyze:
            self.analyze()

    def analyze(self):
        for idx, img in enumerate(self.img_set):
            try:
                log.info(f"Starting plate profiling for plate {idx}")
                plate = PlateProfile(
                    img, self.sample_name,
                    self.n_rows, self.n_cols,
                    self.align, self.fit, True
                )
                plate.add_sampling_day(self.day_index[idx])
                self.plates.append(plate)
            except:
                log.warning(f"Could not analyze plate {idx}: {self.sample_name}", exc_info=True)
                self.invalid_imgs.append(img)
        self.status_analyze = True
        self._compile_series_results()

    def _compile_series_results(self):
        if self.status_analyze is False:
            self.analyze()
        results = []
        for plate in self.plates:
            tmp = plate.get_results()
            tmp = tmp.reset_index(drop=False)
            tmp = tmp.rename(columns={
                "index": "Metric"
            })
            tmp = tmp.set_index(["Metric", "sampling_day"], drop=True)
            results.append(tmp)
        results = pd.concat(results, axis=0, ignore_index=False)
        self.results = results

    def get_results(self):
        return self.results.copy()

    def get_plate_results(self, plate_idx):
        return self.plates[plate_idx].get_results().copy()

    def plot_plate_unfiltered(self, plate_idx, fontsize=34):
        fig, ax = self.plates[plate_idx].plot_unfiltered()
        fig.suptitle(f"{self.sample_name}_plate({plate_idx})", fontsize=fontsize)
        return fig, ax

    def plot_plate_analysis(self, plate_idx, fontsize=34):
        fig, ax = self.plates[plate_idx].plot_analysis()
        fig.suptitle(f"{self.sample_name}_plate({plate_idx})", fontsize=fontsize)
        return fig, ax
