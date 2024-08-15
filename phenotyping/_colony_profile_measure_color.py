from ._colony_profile_measure_base import ColonyProfileMeasureBase


class ColonyProfileMeasureColor(ColonyProfileMeasureBase):

    def _measure_colony(self):
        super()._measure_colony()
        self.measure_color_intensity()

    def measure_color_intensity(self):
        if self.status_object is False:
            self.find_colony()
        colony = self.masked_img.filled(0)
        self._measurements["Intensity_IntegratedColorIntensityRed"] = colony[:, :, 0].sum()
        self._measurements["Intensity_IntegratedColorIntensityGreen"] = colony[:, :, 1].sum()
        self._measurements["Intensity_IntegratedColorIntensityBlue"] = colony[:, :, 2].sum()

        colony = self.background_img.filled(0)
        self._measurements["Intensity_BackgroundIntegratedColorIntensityRed"] = colony[:, :, 0].sum()
        self._measurements["Intensity_BackgroundIntegratedColorIntensityGreen"] = colony[:, :, 1].sum()
        self._measurements["Intensity_BackgroundIntegratedColorIntensityBlue"] = colony[:, :, 2].sum()