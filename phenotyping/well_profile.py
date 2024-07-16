import pandas as pd
import skimage.color as skcolor

import cellprofiler_core as cpc
import cellprofiler.modules as cpm


class WellProfile:
    # Inputs
    input_img = gray_img = None
    plate_idx = None

    img_set_list = None
    img_set = None

    obj_set = None
    cpc_measurements = None
    pipeline = None

    img = None
    img_name = None
    primary_obj_name = "PrimaryObjects"
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
        mod = cpm.identifyprimaryobjects.IdentifyPrimaryObjects()
        mod.x_name.value = self.img_name
        mod.y_name.value = self.primary_obj_name

        self.pipeline.run_module(mod, self.workspace)

    def _run_merge2colony(self, merge_option="Distance",
                          relabel_option="Merge",
                          distance_threshold=0,
                          ref_point="Closest Point"):
        mod = cpm.splitormergeobjects.SplitOrMergeObjects()
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
        self.img = cpc.image.Image(self.gray_img)
        self.img_set.add(self.img_name, self.img)

    def _init_workspace(self):
        '''
        ----- Objective -----
        Initialize CellProfiler Workspace and Identifies Primary Object in Images. This has to be done first before running other modules
        '''
        self.workspace = cpc.workspace.Workspace(
            self.pipeline,
            cpc.module.Module(),
            self.img_set,
            self.obj_set,
            self.cpc_measurements,
            self.img_set_list
        )
        self.status_workspace = True

    def _init_cpc(self):
        counter = 0
        def _sub_init_cpc(counter):
            try:
                img_set_list = cpc.image.ImageSetList()
                img_set = img_set_list.get_image_set(0)

                obj_set = cpc.object.ObjectSet()
                cpc_measurements = cpc.measurement.Measurements()
                pipeline = cpc.pipeline.Pipeline()
            except:
                if counter<50:
                    counter+=1
                    _sub_init_cpc(counter)
            else:
                return img_set_list, img_set, obj_set, cpc_measurements, pipeline
        self.img_set_list, self.img_set, self.obj_set, self.cpc_measurements, self.pipeline = _sub_init_cpc(counter)
