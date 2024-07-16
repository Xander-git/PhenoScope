# ----- Imports -----
import pandas as pd
import skimage.color as skcolor

from cellprofiler_core.preferences import set_headless
from cellprofiler_core.image import ImageSetList, Image
from cellprofiler_core.object import ObjectSet
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.workspace import Workspace
from cellprofiler_core.module import Module

from cellprofiler.modules.identifyprimaryobjects import IdentifyPrimaryObjects
from cellprofiler.modules.splitormergeobjects import SplitOrMergeObjects

set_headless()

# ----- Pkg Relative Import -----

# ----- Main Class Definition -----
class WellProfile:
    # Inputs
    input_img = gray_img = None
    plate_idx = None

    img_set_list = ImageSetList()
    img_set = img_set_list.get_image_set(0)

    obj_set = ObjectSet()
    cpc_measurements = Measurements()
    pipeline = Pipeline()

    img = None
    img_name = None
    primary_obj_name = None
    colony_obj_name = None
    workspace=None

    colony_obj = None

    status_analyis=False
    status_workspace = False

    # TODO: Integrate a settings option
    settings = {

    }
    def __init__(self, well_img, well_name,
                 merge_option="Distance", relabel_option="Merge",
                 distance_threshold=0, ref_point="Closest Point"):
        self.colony_obj_name = f"{well_name}_colony"
        self.primary_obj_name = f"{well_name}_PrimaryObjects"
        self._init_workspace()
        self._set_img(well_img, f"{well_name}")
        self._run_id_primary_obj()
        self._run_merge2colony(
            merge_option, relabel_option, distance_threshold, ref_point
        )

    def _get_metric(self, metric_class, metric_name):
        return self.cpc_measurements.get_measurement(
            self.colony_obj_name,
            f"{metric_class}_{metric_name}_{self.img_name}"
    )

    def _run_id_primary_obj(self):
        mod = IdentifyPrimaryObjects()
        mod.x_name.value = self.img_name

        counter = 2
        while self.primary_obj_name in self.obj_set.get_object_names():
            self.primary_obj_name = f"{self.primary_obj_name}_{counter}"
            counter += 1
        mod.y_name.value = self.primary_obj_name
        self.pipeline.add_module(mod)
        mod.run(self.workspace)

    def _run_merge2colony(self, merge_option="Distance",
                          relabel_option="Merge",
                          distance_threshold=0,
                          ref_point="Closest Point"):
        mod = SplitOrMergeObjects()
        mod.objects_name.value = self.primary_obj_name
        mod.output_objects_name.value = self.colony_obj_name
        mod.merge_option.value = merge_option
        mod.relabel_option.value = relabel_option
        mod.distance_threshold.value = distance_threshold
        mod.where_algorithm.value=ref_point

        self.pipeline.run_module(mod, self.workspace)
        self.colony_obj = self.obj_set.get_objects(
            self.colony_obj_name
        )

    def _set_img(self, well_img, sample_name):
        self.input_img = well_img
        self.img_name = sample_name
        self.gray_img = skcolor.rgb2gray(self.input_img)
        self.img = Image(self.gray_img)
        self.img_set.add(self.img_name, self.img)

    def _init_workspace(self):
        '''
        ----- Objective -----
        Initialize CellProfiler Workspace and Identifies Primary Object in Images. This has to be done first before running other modules
        '''
        self.workspace = Workspace(
            self.pipeline,
            Module(),
            self.img_set,
            self.obj_set,
            self.cpc_measurements,
            self.img_set_list
        )
        self.status_workspace = True

