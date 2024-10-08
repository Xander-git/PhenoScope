import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt


class Image:
    def __init__(self, image: np.ndarray):
        self.__input_image: np.ndarray = image
        self.__image: np.ndarray = image
        self.__object_mask: np.ndarray = None
        self.__object_map: np.ndarray = None

    def __getitem__(self, index):
        return self.__class__(self.__image[index])

    @property
    def image(self):
        return self.__image

    @image.setter
    def image(self, image):
        self.__image = image
        self.__object_mask = None
        self.__object_map = None

    @property
    def object_mask(self):
        return self.__object_mask

    @object_mask.setter
    def object_mask(self, mask):
        self.__object_mask = mask
        self.__object_map = None

    @property
    def object_map(self):
        return self.__object_map

    @object_map.setter
    def object_map(self, map):
        self.__object_map = map

    def imshow(self, cmap='grayscale', figsize=(9, 10)) -> (plt.Figure, plt.Axes):
        fig, ax = plt.subplots()
