import matplotlib.pyplot as plt

from .plate_series_change_over_time import PlateSeriesChangeOverTime

# TODO: Add module logging

class PlateSeries(PlateSeriesChangeOverTime):
    def plot_plate_unfiltered(self, plate_idx, fontsize=34):
        fig, ax = self.plates[plate_idx].plot_unfiltered()
        fig.suptitle(f"{self.sample_name}_plate({plate_idx})", fontsize=fontsize)
        return fig, ax

    def plot_plate_analysis(self, plate_idx, fontsize=34):
        fig, ax = self.plates[plate_idx].plot_analysis()
        fig.suptitle(f"{self.sample_name}_plate({plate_idx})", fontsize=fontsize)
        return fig, ax

    def plot_plate_segmentation(self, plate_idx, figsize=(16, 24)):
        fig, ax = self.plates[plate_idx].plot_analysis_segmentation(
            figsize=figsize
        )
        return fig, ax

    def save_results2csv(self, filepath):
        assert filepath.endswith(".csv")
        self.results.to_csv(filepath)

    def save_analysis_segmentation(self, dirpath, figsize=(16, 24)):
        for plate in self.plates:
            fig, ax = plate.plot_analysis_segmentation(
                figsize=figsize
            )
            fig.savefig(f"{dirpath}{plate.sample_name}.png")
            plt.close(fig)
