import pandas as pd

from cellprofiler.modules.measureobjectsizeshape import MeasureObjectSizeShape

import os
import logging
logger_name = "phenomics-cellprofiler_api"
log = logging.getLogger(logger_name)
logging.basicConfig(format=f'[%(asctime)s|%(levelname)s|{os.path.basename(__file__)}] %(message)s')

# ----- Pkg Relative Import -----
from ._cp_api_object import CellProfilerApiObject

# ----- Global Constants -----
METRIC_LABEL = "Metric"

MAIN_METRICS = [
	""
]

# ----- Main Class Definition -----
# TODO: Cleanup & optimize setting calls
class CellProfilerApiMeasureAreaShape(CellProfilerApiObject):
	status_areashape = True
	areashape_calculate_adv = False
	areashape_calculate_zernikes = False

	def run(self):
		super().run()
		try:
			if self.status_validity and self.status_areashape:
				log.debug(f"Measuring AreaShape for {self.colony_name}")
				self.measure_areashape(self.areashape_calculate_adv, self.areashape_calculate_zernikes)
		except:
			log.warning(f"Could not calculate AreaShape for {self.well_name}", exc_info=True)
			self.status_validity = False

	def measure_areashape(self, calculate_adv=False, calculate_zernike=False):
		MEASUREMENT_CLASS_LABEL = "AreaShape"
		log.debug(f"Measuring Area Shape of sample {self.colony_name}")
		mod = MeasureObjectSizeShape()
		mod.calculate_advanced.value = calculate_adv
		mod.calculate_zernikes.value = calculate_zernike
		mod.objects_list.value = f"{self.colony_name}"
		self.pipeline.run_module(mod, self.workspace)
		keys = self._get_feature_keys(self.colony_name,mod)
		self.keys[f"{MEASUREMENT_CLASS_LABEL}"]=keys
		self._update_results( f"{MEASUREMENT_CLASS_LABEL}",self._get_results(
			self.colony_name,
			keys
		))
		return self.results[f"{MEASUREMENT_CLASS_LABEL}"]
