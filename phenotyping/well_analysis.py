import pandas as pd

import cellprofiler.modules as cpm
import cellprofiler_core as cpc
from typing import Dict, List

from .well_profile import WellProfile

# ----- Global Constats -----
INTEGRATED_INTENSITY = "IntegratedIntensity"
MEAN_INTENSITY = "MeanIntensity"
STD_INTENSITY = "StdIntensity"
MIN_INTENSITY = "MinIntensity"
MAX_INTENSITY = "MaxIntensity"
MEDIAN_INTENSITY = "MedianIntensity"
MAD_INTENSITY = "MADIntensity"
UPPER_QUARTILE_INTENSITY = "UpperQuartileIntensity"
LOWER_QUARTILE_INTENSITY = "LowerQuartileIntensity"


MAIN_METRICS = [
    ""
]
# ----- Main Class Definition -----


class WellAnalysis(WellProfile):
    results = {}

    # Checks to ensure each module only run once
    status_sizeshape = False
    status_intensity = False
    status_texture = False

    def measure_sizeshape(self):
        if self.status_sizeshape is False:
            desired_properties = [
                "label",
                "image",
                "area",
                "perimeter",
                "bbox",
                "bbox_area",
                "major_axis_length",
                "minor_axis_length",
                "orientation",
                "centroid",
                "equivalent_diameter",
                "extent",
                "eccentricity",
                "convex_area",
                "solidity",
                "euler_number",
                "inertia_tensor",
                "inertia_tensor_eigvals",
                "moments",
                "moments_central",
                "moments_hu",
                "moments_normalized",
                "solidity"
            ]
            mod = cpm.measureobjectsizeshape.MeasureObjectSizeShape()
            table = pd.DataFrame(
                mod.analyze_objects(
                    self.colony_obj, desired_properties
                ), index=[self.colony_obj_name]
            )
            table = table.transpose().reset_index(drop=False).rename(
                columns={
                    "index":"Metric"
                }
            )
            table["origin_method"]="SizeShape"
            self.results["sizeshape"] = table
            self.status_sizeshape = True
            return table
        else:
            return self.results["sizeshape"]

    def measure_intensity(self, filter_largest_obj=True):
        '''
        :return:
        '''
        if self.status_intensity is False:
            mod = cpm.measureobjectintensity.MeasureObjectIntensity()
            mod.images_list.value = self.img_name
            mod.objects_list.value = self.colony_obj_name
            self.pipeline.run_module(mod, self.workspace)

            def get_intensity_metric(metric_name):
                return self._get_metric("Intensity", metric_name)

            table = pd.DataFrame({
                "area":self.colony_obj.areas,
                "Total_Intensity": get_intensity_metric(INTEGRATED_INTENSITY),
                "Mean_Intensity": get_intensity_metric(MEAN_INTENSITY),
                "StdDev_Intensity": get_intensity_metric(STD_INTENSITY),
                "Min_Intensity": get_intensity_metric(MIN_INTENSITY),
                "Max_Intensity": get_intensity_metric(MAX_INTENSITY),
                "Median_Intensity": get_intensity_metric(MEDIAN_INTENSITY),
                "MAD_Intensity": get_intensity_metric(MAD_INTENSITY),
                "Upper_Quartile_Intensity": get_intensity_metric(UPPER_QUARTILE_INTENSITY),
                "Lower_Quartile_Intensity": get_intensity_metric(LOWER_QUARTILE_INTENSITY)
            }, index=[self.colony_obj_name])
            if filter_largest_obj:
                table=table[
                    table["area"]==table["area"].max()
                    ]
            table = table[[
                "Total_Intensity", "Mean_Intensity", "StdDev_Intensity",
                "Min_Intensity", "Max_Intensity", "Median_Intensity",
                "MAD_Intensity", "Upper_Quartile_Intensity", "Lower_Quartile_Intensity"
            ]].transpose().reset_index(drop=False).rename(columns={
                "index":"Metric"
            })

            table["origin_method"]= "Intensity"
            self.results["intensity"] = table
            self.status_intensity = True
            return table
        else:
            return self.results["intensity"]

    def measure_texture(self, scale=2, gray_levels=256,
                        metric_axis_map:Dict=None):
        '''
        :param scale:
        :param metric_axis_map:
        :return:
        --- Description ---
        Measurements are computed in 4 different directions:
            00 = Horizontal
            01 = Diagonal (135 deg)
            02 = Vertical (90 deg)
            04 = Diagonal (45 deg)
        '''
        if self.status_intensity is False:
            if metric_axis_map is None:
                metric_axis_map = {
                    "00":"0_deg",
                    "01":"135_deg",
                    "02":"90_deg",
                    "03":"45_deg"
                }
            mod = cpm.measuretexture.MeasureTexture()
            mod.gray_levels.value=gray_levels

            mod.images_or_objects.value = "Objects"
            # May potentially integrate image or object switch in future versions

            mod.add_scale(scale)
            table = pd.DataFrame(
                mod.run_one(self.img_name,
                                        self.colony_obj_name,
                                        scale,
                                        self.workspace),
                columns = ["Img_Name", "Object_Name", "Metric", "Scale", self.colony_obj_name]
            )
            scale_dir = table["Scale"].str.split("_", expand=True)

            dir = scale_dir[1].map(lambda x:metric_axis_map[x])
            table["Metric"] = \
                table["Metric"] + "_" + dir
            table = table.loc[:,["Metric", self.colony_obj_name]]
            scaleframe = pd.DataFrame({
                "Metric": "Texture Scale",
                self.colony_obj_name: scale
            }, index=[0])

            table = pd.concat([table, scaleframe], axis=0).reset_index(drop=True)
            table["origin_method"]="Texture"
            self.results["texture"]=table
            self.status_texture = True
            return table
        else:
            return self.results["texture"]

    def measure_all(self):
        self.measure_sizeshape()
        self.measure_intensity()
        self.measure_texture()

    def get_results(self, metric_names: List[str] = None, measurement_type: str = None):
        if measurement_type is None:
            results = pd.concat(
                list(self.results.values()),
                axis=0
            ).reset_index(drop=True)
            if metric_names is None:
                return results
            else:
                return results[metric_names]
        else:
            return self.results[measurement_type]