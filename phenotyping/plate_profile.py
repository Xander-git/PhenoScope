import pandas as pd
from ..normalization.plate_normalize import PlateNormalize
from .well_analysis import WellAnalysis
from typing import List
import logging
class PlateProfile:
	sample_name = None
	plate = None
	wells = []
	bad_well_imgs = []
	def __init__(self, img, sample_name,
				 auto_analyze:bool=False,
				 n_rows:int=8, n_cols:int=12,
				 blob_detection_method:str='log',
				 min_sigma:int=4, max_sigma:int=50, num_sigma:int=30,
				 threshold:float=0.01, overlap:float=0.1, min_size:int=180,
				 border_padding:int=50, verbose=True,
				 ):
		self.sample_name=sample_name
		self.plate = PlateNormalize(
			img, n_rows, n_cols,
			blob_detection_method,
			min_sigma, max_sigma, num_sigma,
			threshold, overlap, min_size,
			border_padding, verbose
		)
		if auto_analyze:
			self.generate_well_profiles()

	def generate_well_profiles(
			self, auto_measure=True,
			merge_option="Distance", relabel_option="Merge",
			distance_threshold=0, ref_point="Closest Point"
	):
		well_imgs = self.plate.get_well_imgs()
		for idx, well_img in enumerate(well_imgs):
			print(f"Analyzing well {idx}")
			try:
				well_profile = WellAnalysis(well_img, f"{self.sample_name}_well_{idx}",
											merge_option, relabel_option,
											distance_threshold, ref_point
											)
				if auto_measure:
					well_profile.measure_all()
				self.wells.append(well_profile)
			except:
				logging.error(f"Could not analyze well {idx}", exc_info=True)
				self.bad_well_imgs.append(idx)

	def get_well_results(self, metric_names: List[str]=None, measurement_type: str = None):
		well_results = []
		for well in self.wells:
			well_results.append(well.get_results(metric_names, measurement_type))
		well_results = pd.concat(well_results, axis=1)
		return well_results

	def run_well_analysis(self):
		for well in self.wells:
			well.measure_all()