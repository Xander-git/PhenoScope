# ----- Imports -----
import numpy as np
import skimage.color as skcolor
import pandas as pd

from cellprofiler_core.preferences import set_headless
from cellprofiler_core.image import ImageSetList, Image
from cellprofiler_core.object import ObjectSet
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.workspace import Workspace
from cellprofiler_core.module import Module

import os
import logging
logger_name = "phenomics-cellprofiler_api"
log = logging.getLogger(logger_name)
logging.basicConfig(format=f'[%(asctime)s|%(levelname)s|{os.path.basename(__file__)}] %(message)s')

# ----- Pkg Relative Import -----

# ----- Main Class Definition -----
set_headless()


class CellProfilerApiBase:
	# Inputs
	input_img = gray_img = None
	plate_idx = None

	img_set_list = None
	img_set = None

	obj_set = None
	cpc_measurements = None
	pipeline = None

	img = None

	well_name = None
	primary_name = None
	merged_name = None
	filled_name = None
	colony_name = None
	noise_name = None
	workspace = None

	colony = None

	keys = {}
	results = {}
	results_table = None

	status_validity = True

	status_workspace = False

	# TODO: Integrate a settings option
	settings = {

	}

	def __init__(self, well_img: np.ndarray, well_name: str, auto_run=False):
		logging.info(f"Starting Well Analysis for: {well_name}")
		self._set_name(well_name)
		self._set_img(well_img)
		if auto_run is True:
			self.run()

	def run(self):
		try:
			self._init_workspace()
		except:
			log.warning(f"Could not intitialize workspace for {self.well_name}", exc_info=True)
			self.status_validity = False

	def _set_name(self, well_name):
		self.well_name = well_name
		self.primary_name = f"{well_name}_PrimaryObjects"
		self.merged_name = f"{well_name}_MergedObjects"
		self.filled_name = f"{well_name}_FilledObjects"
		self.noise_name = f"{well_name}_Noise"
		self.colony_name = f"{well_name}_Colony"

	def _set_img(self, well_img):
		self.input_img = well_img
		self.gray_img = skcolor.rgb2gray(self.input_img)

	def _init_workspace(self):
		'''
		----- Objective -----
		Initialize CellProfiler Workspace and Identifies Primary Object in Images. This has to be done first before running other modules
		'''
		self.img_set_list = ImageSetList()
		self.img_set = self.img_set_list.get_image_set(0)

		self.obj_set = ObjectSet(can_overwrite=True)

		self.cpc_measurements = Measurements()

		self.pipeline = Pipeline()

		self.img = Image(self.gray_img)
		self.img_set.add(self.well_name, self.img)

		self.workspace = Workspace(
			self.pipeline,
			Module(),
			self.img_set,
			self.obj_set,
			self.cpc_measurements,
			self.img_set_list
		)
		self.status_workspace = True

	def _get_feature_keys(self, obj_name, module):
		cpc_metric_cols = pd.DataFrame(
			module.get_measurement_columns(self.pipeline),
			columns=["source", "keys", "dtype"]
		)
		keys = cpc_metric_cols[cpc_metric_cols["source"] == obj_name]
		return keys.loc[:, "keys"]

	def _get_results(self, obj_name, feature_keys):
		results = []
		for key in feature_keys:
			curr_result = self.cpc_measurements.get_measurement(
				obj_name,
				key
			)
			assert (len(curr_result) == 1), "Too many objects"
			results.append(curr_result[0])
		results = pd.DataFrame(
			list(zip(feature_keys, results)),
			columns=["Metric", obj_name]
		).set_index("Metric").copy(deep=True)
		log.debug(f"Got results for {obj_name} resulting shape: {results.shape}")
		return results

	def _update_results(self, metric_name: str, data):
		self.results[metric_name] = data.copy()
