from typing import List

import numpy as np

from ._object_finder_base import ObjectFinderBase


class ObjectFinderMeasurementFilter(ObjectFinderBase):
    def __init__(
            self,
            threshold_method: str = "otsu",
            enhance_contrast: bool = True,
            max_eccentricity: float = 0.85,
            min_circularity: float = 0.7,
            min_area: int = None,
            **kwargs
    ):
        super().__init__(
                threshold_method=threshold_method,
                enhance_contrast=enhance_contrast,
                **kwargs
        )
        self._max_eccentricity = max_eccentricity
        self._min_circularity = min_circularity
        self._min_area = min_area

    def find_objects(self, image: np.ndarray):
        super().find_objects(image=image)
        self.filter_objects()

    def filter_objects(self):
        self._eccentricity_filter()
        self._circularity_filter()
        self._area_filter()

    def _eccentricity_filter(self):
        self._table = self._table.loc[
                      self._table.loc[:, "eccentricity"] < self._max_eccentricity, :
                      ]

    def _circularity_filter(self):
        self._table = self._table.loc[
                      self._table.loc[:, "circularity"] > self._min_circularity, :
                      ]

    def _area_filter(self):
        if self._min_area is None:
            self._min_area = int((min(self._img_dims) * 0.01) ** 2)

        self._table = self._table.loc[
                      self._table.loc[:, "area"] > self._min_area, :
                      ]
