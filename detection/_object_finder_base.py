from typing import List

import pandas as pd
import numpy as np
from skimage.measure import regionprops_table
from skimage.filters import threshold_triangle, threshold_otsu
from skimage.measure import label
from skimage.morphology import white_tophat, square

from .clahe_boost import ClaheBoost
from ..util import check_grayscale
class ObjectFinderBase:
    def __init__(self, img: np.ndarray, boost_segmentation: bool = True):
        self.__input_img = check_grayscale(img)
        self._boost_segmentation = boost_segmentation

        self._threshold_method = None

        self._obj_map = None
        self._table = pd.DataFrame()

        self._essential_measurements = [
            "label",
            "centroid",
            "bbox",
            "area_bbox",
            "eccentricity",
            "intensity_mean",
        ]

        self._basic_measurements = [
            "area",
            "area_bbox",
            "area_convex",
            "area_filled",
            "equivalent_diameter_area",
            "intensity_max",
            "solidity",
            "perimeter",
            "orientation",
            "num_pixels",
            "extent"
        ]

    @property
    def input_img(self):
        return self.__input_img

    @property
    def results(self):
        if self._table.empty: self.find_objects()
        return self._table.copy(deep=True).set_index("label")

    def get_results(self):
        return self.results

    def find_objects(self, threshold_method: str = "otsu",
                     measurements: str or List[str] = "basic",
                     **kwargs
                     ):
        self._segment_objects(
                threshold_method=threshold_method,
                **kwargs
        )
        self._measure_objects(measurements=measurements)

    def _segment_objects(self, threshold_method="otsu", **kwargs):
        if self._boost_segmentation:
            img = ClaheBoost(
                    img=self.input_img,
                    footprint_shape=kwargs.get("footprint_shape", "square"),
                    footprint_radius=kwargs.get("footprint_radius", None),
                    kernel_size=kwargs.get("kernel_size", None),
            ).get_boosted_img()
        else:
            img = self.input_img

        self._threshold_method = threshold_method
        if self._threshold_method == "otsu":
            thresh = threshold_otsu(img)
        elif self._threshold_method == "triangle":
            thresh = threshold_triangle(img)
        else:
            thresh = threshold_otsu(img)

        binary_img = img > thresh
        res = white_tophat(
                image=binary_img,
                footprint=square(width=int(min(self.input_img.shape) * 0.01))
        )
        self._obj_map = label(binary_img & ~res)

    def _measure_objects(self,
                         measurements: str or List[str] = "basic"
                         ):
        region_props = self._essential_measurements
        if measurements is None:
            pass
        elif measurements == "basic":
            region_props += self._basic_measurements
        elif type(measurements) == List[str]:
            region_props += measurements
        else:
            ValueError("Invalid input for measurement")

        self._table = pd.DataFrame(data=regionprops_table(
                label_image=self._obj_map,
                intensity_image=self.input_img,
                properties=region_props,
                cache=False
        ))
        self._table = self._table.rename(columns={
            "bbox-0"    : "bbox-row_min",
            "bbox-1"    : "bbox-col_min",
            "bbox-2"    : "bbox-row_max",
            "bbox-3"    : "bbox-col_max",
            "centroid-0": "centroid-row",
            "centroid-1": "centroid-col"
        })
