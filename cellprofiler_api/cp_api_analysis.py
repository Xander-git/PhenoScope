import pandas as pd
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt

from typing import List

import logging

import os
import logging
logger_name = "phenomics-cellprofiler_api"
log = logging.getLogger(logger_name)
logging.basicConfig(format=f'[%(asctime)s|%(levelname)s|{os.path.basename(__file__)}] %(message)s')
# ----- Pkg Relative Import -----
from .cp_api_measure_texture import CellProfilerApiMeasureTexture

# ----- Global Constants -----
METRIC_LABEL = "Metric"


# ----- Main Class Definition -----


class CellProfilerApiAnalysis(CellProfilerApiMeasureTexture):
	# Checks to ensure each module only run once
	results_table = None

	# def run(self):
	# 	super().run()
	# 	try:
	# 		self._compile_results()
	# 		results = {
	# 			"results"                : self.results_table.copy(),
	# 			"segmentation"           : np.copy(self.segmentation),
	# 			"unfiltered_segmentation": np.copy(self.unfiltered_segmentation),
	# 			"status_validity"        : self.status_validity
	# 		}
	# 		self.workspace.close()
	# 		return results
	# 	except:
	# 		log.warning(f"Could not compile results for {self.well_name}")
	# 	finally:
	# 		self.workspace.close()

	def run(self):
		Results = namedtuple("Results",["results","segmentation", "unfiltered_segmentation", "status_validity"])

		try:
			self._init_workspace()
		except:
			log.warning(f"Could not intitialize workspace for {self.well_name}", exc_info=True)
			self.status_validity = False

		try:
			if self.status_validity:
				log.debug(f"Identifying & Filtering Image Objects for {self.well_name}")
				self._run_id_primary_obj()
				self._run_merge_obj()
				filled_segmentation = self._run_fill_obj()
				segmentation = self._run_filter_obj()
		except:
			log.warning(f"Could not identify objects and filter colony in {self.well_name}", exc_info=True)
			self.status_validity = False

		try:
			if self.status_validity and self.status_areashape:
				log.debug(f"Measuring AreaShape for {self.colony_name}")
				self.measure_areashape(self.areashape_calculate_adv, self.areashape_calculate_zernikes)
		except:
			log.warning(f"Could not calculate AreaShape for {self.well_name}", exc_info=False)
			self.status_validity = False

		try:
			if self.status_validity and self.status_intensity:
				log.debug(f"Measuring intensity for {self.colony_name}")
				self.measure_intensity()
		except:
			log.warning(f"Could not measure intensity for {self.well_name}", exc_info=False)
			self.status_validity = False

		try:
			if self.status_validity and self.status_texture:
				log.debug(f"Measuring texture for {self.colony_name}")
				self.measure_texture(self.texture_scale, self.texture_gray_lvls)
		except:
			log.warning(f"could not measure texture for {self.well_name}", exc_info=False)
			self.status_validity = False

		try:
			self._compile_results()
			results = Results(self.results_table.copy(), segmentation, filled_segmentation, self.status_validity)
			# results = {
			# 	"results"                : self.results_table.copy(),
			# 	"segmentation"           : segmentation,
			# 	"unfiltered_segmentation": filled_segmentation,
			# 	"status_validity"        : self.status_validity
			# }
			self.workspace.close()
			return results
		except:
			log.warning(f"Could not compile results for {self.well_name}")
		finally:
			self.workspace.close()


	def get_results(self):
		# Deprecated when module was converted to an API instead of integration
		return self.results_table.copy()

	def _compile_results(self):
		validity = pd.DataFrame({
			f"{self.colony_name}": self.status_validity,
		}, index=["status_validity"]
		).astype(int)
		self.results_table = pd.concat(
			self.results.values(), axis=0
		)
		self.results_table = pd.concat([self.results_table, validity], axis=0)
