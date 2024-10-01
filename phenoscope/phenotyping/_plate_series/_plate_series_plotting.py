from ._plate_series_growth import PlateSeriesGrowth


# TODO: Add module logging

class PlateSeriesPlotting(PlateSeriesGrowth):
    def plot_plate_unfiltered(self, plate_idx, fontsize=34):
        fig, ax = self.plates[plate_idx].plot_unfiltered()
        fig.suptitle(f"{self.series_name}_plate({plate_idx})", fontsize=fontsize)
        return fig, ax

    def plot_plate_analysis(self, plate_idx, fontsize=34):
        fig, ax = self.plates[plate_idx].plot_analysis()
        fig.suptitle(f"{self.series_name}_plate({plate_idx})", fontsize=fontsize)
        return fig, ax

    def plot_plate_segmentation(self, plate_idx, figsize=(16, 24)):
        fig, ax = self.plates[plate_idx].plot_analysis_segmentation(
                figsize=figsize
        )
        return fig, ax

    def plot_plate_colony_segmentation(self, plate_idx, figsize=(18, 8)):
        fig, ax = self.plates[plate_idx].plot_colony_segmentation(
                figsize=figsize
        )
        return fig, ax
