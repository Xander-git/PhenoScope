import numpy as np
import matplotlib.pyplot as plt

from cellprofiler.modules.identifyprimaryobjects import IdentifyPrimaryObjects
from cellprofiler.modules.splitormergeobjects import SplitOrMergeObjects
from cellprofiler.modules.fillobjects import FillObjects
from cellprofiler.modules.measureobjectsizeshape import MeasureObjectSizeShape
from cellprofiler.modules.filterobjects import FilterObjects

from .cp_api_base import CellProfilerApiBase

import os
import logging
logger_name = "phenomics-cellprofiler_api"
log = logging.getLogger(logger_name)
logging.basicConfig(format=f'[%(asctime)s|%(levelname)s|{os.path.basename(__file__)}] %(message)s')


class CellProfilerApiFilter(CellProfilerApiBase):
	merge_option = "Distance"
	relabel_option = "Merge"
	distance_threshold = 0
	ref_point = "Closest Point"
	max_hole_size = 64
	segmentation = None
	unfiltered_segmentation = None

	def run(self):
		super().run()
		try:
			if self.status_validity:
				log.info(f"Identifying & Filtering Image Objects for {self.well_name}")
				self._run_find_colony()
		except:
			log.warning(f"Could not identify objects and filter colony in {self.well_name}", exc_info=True)
			self.status_validity = False

	def _run_find_colony(self):
		self._run_id_primary_obj()
		self._run_merge_obj()
		self._run_fill_obj()
		self._run_filter_obj()

	def _run_id_primary_obj(self):
		mod = IdentifyPrimaryObjects()
		mod.x_name.value = self.well_name
		mod.y_name.value = self.primary_name

		self.pipeline.add_module(mod)
		mod.run(self.workspace)

	def _run_merge_obj(self):
		mod = SplitOrMergeObjects()
		mod.objects_name.value = self.primary_name
		mod.output_objects_name.value = self.merged_name
		mod.merge_option.value = self.merge_option
		mod.relabel_option.value = self.relabel_option
		mod.distance_threshold.value = self.distance_threshold
		mod.where_algorithm.value = self.ref_point

		self.pipeline.run_module(mod, self.workspace)

	def _run_fill_obj(self):
		mod = FillObjects()
		mod.x_name.value = self.merged_name
		mod.y_name.value = self.filled_name
		mod.size.value = self.max_hole_size
		mod.mode.value = "Holes"
		self.pipeline.add_module(mod)
		mod.run(self.workspace)
		obj = self.obj_set.get_objects(self.filled_name)
		self.unfiltered_segmentation = obj.segmented
		return np.copy(obj.segmented)

	def _run_filter_obj(self):
		log.debug("Measuring FilledObject AreaShape")
		sub_mod = MeasureObjectSizeShape()
		sub_mod.calculate_advanced.value = False
		sub_mod.calculate_zernikes.value = False
		sub_mod.objects_list.value = self.filled_name
		self.pipeline.run_module(sub_mod, self.workspace)

		log.debug("Filtering Objects based on AreaShape_Area")
		mod = FilterObjects()
		mod.x_name.value = self.filled_name
		mod.y_name.value = self.colony_name
		mod.mode.value = "Measurements"
		mod.filter_choice.value = "Maximal"
		mod.keep_removed_objects.value = True
		mod.removed_objects_name.value = self.noise_name
		mod.add_measurement()
		self.pipeline.run_module(mod, self.workspace)
		self.colony = self.obj_set.get_objects(self.colony_name)
		self.segmentation = self.colony.segmented
		return np.copy(self.colony.segmented)

