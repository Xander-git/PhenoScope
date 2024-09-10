from typing import List

import pandas as pd
import numpy as np
from skimage.measure import regionprops_table
from skimage.filters import threshold_triangle, threshold_otsu
from skimage.measure import label
from skimage.morphology import white_tophat, disk

from .clahe_boost import ClaheBoost
from ..util import check_grayscale


class ObjectFinderBase:
    def __init__(
            self,
            threshold_method="otsu",
            measurements: List[str] = None,
            enhance_contrast: bool = True,
            **kwargs
    ):

        self._enhance_contrast = enhance_contrast
        self._footprint_shape = kwargs.get("footprint_shape", "disk")
        self._footprint_radius = kwargs.get("footprint_radius") # Returns none by default if none specified
        self._kernel_size = kwargs.get("kernel_size")

        self._threshold_method = threshold_method

        self._img_dims = None
        self._obj_map = None
        self._table = pd.DataFrame()

        self._essential_measurements = [
            "label",
            "centroid",
            "bbox",
            "area_bbox",
            "area",
            "perimeter",
            "eccentricity",
            "intensity_mean",
        ]

        basic_measurements = [
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

        if measurements is None:
            self._measurements = self._essential_measurements + basic_measurements
        elif type(measurements) == List[str]:
            self._measurements = self._essential_measurements + measurements
        else:
            raise ValueError("Measurements must be a list of strings")

    @property
    def results(self) -> pd.DataFrame:
        if self._table.empty: raise AttributeError(
                "Failed to get results. Try running find_objects() on an image first.")
        return self._table.copy(deep=True).set_index("label")

    def get_results(self) -> pd.DataFrame:
        return self.results

    def find_objects(self, image: np.ndarray) -> None:
        image = check_grayscale(image)
        self._img_dims = image.shape

        self._segment_objects(image=image)
        self._measure_objects(image=image)

    def _segment_objects(self, image: np.ndarray) -> None:
        image = check_grayscale(image)
        if self._enhance_contrast:
            image = ClaheBoost(
                    img=image,
                    footprint_shape=self._footprint_shape,
                    footprint_radius=self._footprint_radius,
                    kernel_size=self._kernel_size,
            ).get_boosted_img()

        if self._threshold_method == "otsu":
            thresh = threshold_otsu(image)
        elif self._threshold_method == "triangle":
            thresh = threshold_triangle(image)
        else:
            thresh = threshold_otsu(image)

        binary_img = image > thresh
        res = white_tophat(
                image=binary_img,
                footprint=disk(radius=int(min(image.shape) * 0.01))
        )
        self._obj_map = label(binary_img & ~res)

    def _measure_objects(self, image: np.ndarray) -> None:

        self._table = pd.DataFrame(data=regionprops_table(
                label_image=self._obj_map,
                intensity_image=image,
                properties=self._measurements,
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

        cols = self._table.columns.tolist()
        cols.insert(0, cols.pop(cols.index("centroid-row")))
        cols.insert(0, cols.pop(cols.index("centroid-col")))
        cols.insert(0, cols.pop(cols.index("label")))
        self._table = self._table.loc[:, cols]

        self._table.insert(
                loc=4,
                column="circularity",
                value=(4 * np.pi * self._table.loc[:, "area"]) / (self._table.loc[:, "perimeter"] ** 2)
        )

