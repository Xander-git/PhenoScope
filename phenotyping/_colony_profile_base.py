import logging
import pandas as pd

formatter = logging.Formatter(
    fmt=f'[%(asctime)s|%(name)s] %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S'
)
console_handler = logging.StreamHandler()
log = logging.getLogger(__name__)
log.addHandler(console_handler)
console_handler.setFormatter(formatter)

import matplotlib.pyplot as plt
from skimage.color import rgb2gray


class ColonyProfileBase:
    def __init__(self, img, sample_name,
                 auto_run=True
                 ):
        self._input_img = (img, None)
        self._sample_name = (sample_name, None)

        self.status_validity = True
        self.status_analysis = False
        if auto_run:
            self.run_analysis()

    @property
    def input_img(self):
        return self._input_img[0]

    @property
    def sample_name(self):
        return self._sample_name[0]

    @property
    def colony_name(self):
        return f"{self.sample_name}_Colony"

    @property
    def gray_img(self):
        if len(self.input_img.shape)==3:
            return rgb2gray(self.input_img)
        elif len(self.input_img.shape)==2:
            return self.input_img
        else:
            raise ValueError('Input image is not grayscale or RGB')

    @property
    def results(self):
        if self.status_analysis is False:
            self.run_analysis()
        if self.status_validity is True:
            return self._results.copy()
        else:
            return pd.Series(
                data=[False],
                index=["status_valid_analysis"],
                name=f"{self.colony_name}"
            )

    @property
    def _results(self):
        pass

    def run_analysis(self):
        pass

    def get_results(self):
        """
        Convienence method for accessing the results table of the colony profile
        :return: self.results
        """
        return self.results
