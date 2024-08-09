import logging

formatter = logging.Formatter(
    fmt=f'[%(asctime)s|%(name)s] %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S'
)
console_handler = logging.StreamHandler()
log = logging.getLogger(__name__)
log.addHandler(console_handler)
console_handler.setFormatter(formatter)

import pandas as pd
import numpy as np
import numpy.ma as ma
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_otsu, threshold_triangle
from skimage.morphology import white_tophat, disk
from skimage.measure import label, regionprops_table

from ._colony_profile_base import ColonyProfileBase
from ..detection import ClaheBoost


class ColonyProfileObject(ColonyProfileBase):
    def __init__(self, img, sample_name, auto_run=True):
        self.thresh = None
        self.segmentation = None
        self.labeled_segmentation = None
        self.colony_mask = None
        self.segment_properties = None
        
        self.colony = None
        self.status_object = False
        super().__init__(img=img, sample_name=sample_name, auto_run=auto_run)


    # TODO: Add a way to finetune boosted_img parameters
    @property
    def boosted_img(self):
        """
        ClaheBoost parameters are optimized for an image of a single well
        :return:
        """
        return ClaheBoost(self.gray_img, kernel_size=None, footprint_radius=5).get_boosted_img()

    @property
    def masked_img(self):
        if self.status_object is False or self.colony_mask is None: self.find_colony()
        dim3_mask = np.expand_dims(np.invert(self.colony_mask), axis=2)
        dim3_mask = np.repeat(dim3_mask, 3, axis=2)
        return ma.array(data=self.input_img, mask=dim3_mask)

    @property
    def masked_gray_img(self):
        if self.status_object is False or self.colony_mask is None: self.find_colony()
        return ma.array(self.input_img, mask=np.invert(self.colony_mask))

    def find_colony(self, threshold_method="otsu", use_boosted=True,
                    filter_property="distance_from_center", filter_type="min",
                    **kwargs
                    ):
        if self.status_validity:
            try:
                self._find_colony(threshold_method=threshold_method, use_boosted=use_boosted,
                                  filter_property=filter_property, filter_type=filter_type,
                                  kwargs=kwargs)
            except:
                log.warning(f"Failed to find colony for {self.sample_name}")
                self.status_validity = False


    def _find_colony(self, threshold_method="otsu", use_boosted=True,
                    filter_property="distance_from_center", filter_type="min",
                    **kwargs
                    ):

        self.find_objects(threshold_method=threshold_method, use_boosted=use_boosted)
        self.fill_object_holes(
            hole_radius=kwargs.get("hole_radius", 10)
        )
        self.filter_particles(
            particle_radius=kwargs.get("particle_radius", None)
        )
        self.filter_properties(property=filter_property, filter_type=filter_type)

    def find_objects(self, threshold_method="otsu", use_boosted=True):
        """
        Creates an initial binary mask of the objects within an image.
        :param use_boosted:
        :return:
        """
        if use_boosted is True:
            img = self.boosted_img
        else:
            img = self.gray_img

        if threshold_method=="otsu":
            self.thresh = threshold_otsu(img)
        elif threshold_method=="triangle":
            self.thresh = threshold_triangle(img)
        else:
            raise ValueError("Unknown threshold method for finding objects")

        self.segmentation = img>self.thresh
        self.status_object = True

    def fill_object_holes(self, hole_radius=10):
        structure = disk(hole_radius)
        self.segmentation = binary_fill_holes(input=self.segmentation,
                                              structure=structure)

    def filter_particles(self, particle_radius: int = None):
        """
        For the particle filter, particles are assumed to be any object
        that has a radius equal to 1/20th the largest side of the image.
        :param particle_radius:
        :return:
        """
        if self.status_object is False:
            self.find_objects()
        if particle_radius is None:
            particle_radius = np.round(max(self.segmentation.shape) / 20)
        footprint = disk(particle_radius)
        tophat_results = white_tophat(self.segmentation, footprint=footprint)
        self.segmentation = self.segmentation & ~tophat_results

    def filter_properties(self, property="distance_from_center", filter_type="min"):
        """
        Filters the identified objects in the image based on the property inputted. Currently implemented filters
        are 'area' and 'distance_from_center'.
        :param property:
        :return:
        """
        self.labeled_segmentation = label(self.segmentation)
        self.segment_properties = pd.DataFrame(
            regionprops_table(label_image=self.labeled_segmentation,
                              intensity_image=self.input_img,
                              properties=[
                                  "label",
                                  "area",
                                  "centroid"
                              ])
        )
        self.segment_properties = self.segment_properties.set_index("label")
        center_row = self.labeled_segmentation.shape[0] / 2
        center_col = self.labeled_segmentation.shape[1] / 2
        self.segment_properties["distance_from_center"] = np.sqrt(
            (self.segment_properties["centroid-0"] - center_row) ** 2 + (self.segment_properties["centroid-1"] - center_col) ** 2
        )
        if filter_type=="min":
            self.colony_mask = self.labeled_segmentation==self.segment_properties[f"{property}"].idxmin()
        elif filter_type=="max":
            self.colony_mask = self.labeled_segmentation==self.segment_properties[f"{property}"].idxmax()
        else:
            raise ValueError("Invalid filter_type. Currently implemented filter types are 'min' or 'max'")

