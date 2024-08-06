import pandas as pd

from cellprofiler.modules.measureobjectintensity import MeasureObjectIntensity

import os
import logging
logger_name = "phenomics-cellprofiler_api"
log = logging.getLogger(logger_name)
logging.basicConfig(format=f'[%(asctime)s|%(levelname)s|{os.path.basename(__file__)}] %(message)s')
# ----- Pkg Relative Import -----
from .cp_api_measure_areashape import CellProfilerApiMeasureAreaShape

# ----- Global Constants -----
INTEGRATED_INTENSITY = "IntegratedIntensity"
MEAN_INTENSITY = "MeanIntensity"
STD_INTENSITY = "StdIntensity"
MIN_INTENSITY = "MinIntensity"
MAX_INTENSITY = "MaxIntensity"
MEDIAN_INTENSITY = "MedianIntensity"
MAD_INTENSITY = "MADIntensity"
UPPER_QUARTILE_INTENSITY = "UpperQuartileIntensity"
LOWER_QUARTILE_INTENSITY = "LowerQuartileIntensity"

METRIC_LABEL = "Metric"


# ----- Main Class Definition -----
# Cleanup & optimize setting calls
class CellProfilerApiMeasureIntensity(CellProfilerApiMeasureAreaShape):
	status_intensity = True

	def run(self):
		super().run()
		try:
			if self.status_validity and self.status_intensity:
				log.debug(f"Measuring intensity for {self.colony_name}")
				self.measure_intensity()
		except:
			log.warning(f"Could not measure intensity for {self.well_name}", exc_info=True)
			self.status_validity = False

	def measure_intensity(self):
		'''
		:return:
		'''
		MEASUREMENT_CLASS_LABEL = "Intensity"
		mod = MeasureObjectIntensity()
		mod.images_list.value = self.well_name
		mod.objects_list.value = self.colony_name
		self.pipeline.add_module(mod)
		mod.run(self.workspace)
		keys = self._get_feature_keys(self.colony_name,mod)
		self.keys[f"{MEASUREMENT_CLASS_LABEL}"] = keys
		results = self._get_results(
			self.colony_name, keys
		).reset_index(drop=False)

		metadata = results["Metric"].str.replace(self.well_name, "source")
		metadata = metadata.str.split("_", expand=True)
		metadata = metadata.rename(
			columns={
				0: "origin_method",
				1: "feature",
				2: "meta1",
				3: "meta2"
			}
		)
		metadata1 = metadata[metadata["origin_method"] == "Intensity"]
		metadata1 = metadata1["origin_method"] + "_" \
					+ metadata1["feature"]

		metadata2 = metadata[metadata["origin_method"] == "Location"]
		metadata2 = metadata2["origin_method"] + "_" \
					+ metadata2["feature"] + "_" \
					+ metadata2["meta1"]
		metadata = pd.concat([metadata1, metadata2], axis=0)
		results["Metric"] = metadata
		self._update_results(f"{MEASUREMENT_CLASS_LABEL}", results.set_index("Metric"))
		return self.results[f"{MEASUREMENT_CLASS_LABEL}"]
