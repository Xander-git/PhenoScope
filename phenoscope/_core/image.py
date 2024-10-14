from typing import Union

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label

from ..util.type_checks import is_binary_mask


class Image:
    def __init__(self, image: np.ndarray):
        self.__image_array: np.ndarray = image
        self.__enhanced_image_array = image
        self.__object_mask: np.ndarray = None
        self.__object_map: np.ndarray = None

    def __getitem__(self, index):
        return self.__class__(self.__image_array[index])

    @property
    def array(self) -> np.ndarray:
        return np.copy(self.__image_array)

    @array.setter
    def array(self, image: np.ndarray) -> None:
        self.__image_array = image
        self.__enhanced_image_array = image
        self.__object_mask = None
        self.__object_map = None

    @property
    def enhanced_array(self) -> np.ndarray:
        return np.copy(self.__enhanced_image_array)

    @enhanced_array.setter
    def enhanced_array(self, enhanced_image: np.ndarray) -> None:
        self.__enhanced_image_array = enhanced_image
        self.__object_mask = None
        self.__object_map = None

    @property
    def object_mask(self) -> np.ndarray:
        return np.copy(self.__object_mask)

    @object_mask.setter
    def object_mask(self, mask: np.ndarray) -> None:
        if is_binary_mask(mask) is False:
            if mask is not None:
                raise ValueError("Mask must be a binary array")
        self.__object_mask = mask
        self.__object_map = label(self.__object_mask)

    @property
    def object_map(self) -> np.ndarray:
        return np.copy(self.__object_map)

    @object_map.setter
    def object_map(self, object_map: np.ndarray) -> None:
        self.__object_map = object_map

    def copy(self):
        new_image = self.__class__(self.array)
        new_image.enhanced_array = self.enhanced_array
        new_image.object_mask = self.object_mask
        new_image.object_map = self.object_map
        return new_image

    def show(self, ax=None, cmap='gray', figsize=(9, 10)) -> (plt.Figure, plt.Axes):
        if ax is None:
            fig, ax = plt.subplots(tight_layout=True, figsize=figsize)

        ax.grid(False)
        if len(self.array.shape) == 2:
            ax.imshow(self.array, cmap=cmap)
        else:
            ax.imshow(self.array)

        if ax is None:
            return fig, ax
        else:
            return ax

    def show_enhanced(self, ax=None, cmap='gray', figsize=(9, 10)) -> (plt.Figure, plt.Axes):
        if ax is None:
            fig, ax = plt.subplots(tight_layout=True, figsize=figsize)

        ax.grid(False)
        if len(self.enhanced_array.shape) == 2:
            ax.imshow(self.enhanced_array, cmap=cmap)
        else:
            ax.imshow(self.enhanced_array)

        if ax is None:
            return fig, ax
        else:
            return ax

    def show_mask(self, ax=None, cmap='gray', figsize=(9, 10)) -> (plt.Figure, plt.Axes):
        if ax is None:
            fig, ax = plt.subplots(tight_layout=True, figsize=figsize)

        ax.grid(False)
        if len(self.object_mask.shape) == 2:
            ax.imshow(self.object_mask, cmap=cmap)
        else:
            ax.imshow(self.object_mask)

        if ax is None:
            return fig, ax
        else:
            return ax

    def show_map(self, ax=None, cmap='gray', figsize=(9, 10)) -> (plt.Figure, plt.Axes):
        if ax is None:
            fig, ax = plt.subplots(tight_layout=True, figsize=figsize)

        ax.grid(False)
        if len(self.object_map.shape) == 2:
            ax.imshow(self.object_map, cmap=cmap)
        else:
            ax.imshow(self.object_map)

        if ax is None:
            return fig, ax
        else:
            return ax
