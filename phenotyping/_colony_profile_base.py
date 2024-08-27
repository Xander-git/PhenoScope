import logging

import numpy as np
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
    def __init__(self, img: np.ndarray, image_name: str,
                 auto_run: bool = True
                 ):
        self.__input_img = img
        self.__image_name = image_name

        self.status_validity = True
        self.status_analysis = False
        if auto_run:
            self.run()

    @property
    def input_img(self):
        return self.__input_img

    @property
    def image_name(self):
        return self.__image_name

    @property
    def colony_name(self):
        return f"{self.image_name}_Colony"

    @property
    def background_name(self):
        return f"{self.image_name}_Background"

    @property
    def gray_img(self):
        if len(self.input_img.shape) == 3:
            return rgb2gray(self.input_img)
        elif len(self.input_img.shape) == 2:
            return self.input_img
        else:
            raise ValueError('Input image is not grayscale or RGB')

    @property
    def results(self):
        if self.status_analysis is False:
            self.run()
        # if self.status_validity is True:
        results = self._results
        if self.status_validity is False or self._results.empty:
            results.loc["status_valid_analysis"] = 0
        else:
            results.loc["status_valid_analysis"] = self.status_validity
        return results

    @property
    def _results(self):
        """
        Placeholder for future work
        :return:
        """
        return pd.Series()

    def run(self):
        pass

    def get_results(self):
        """
        Convienence method for accessing the results table of the colony profile
        :return: pd.DataFrame
        """
        return self.results
