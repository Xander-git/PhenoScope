import matplotlib.pyplot as plt

from ._plate_series_plotting import PlateSeriesPlotting

class PlateSeriesIO(PlateSeriesPlotting):
    def save_results2csv(self, filepath):
        assert filepath.endswith(".csv")
        self.results.to_csv(filepath, header=False)

    def save_analysis_segmentation(self, dirpath, figsize=(16, 24)):
        for plate in self.plates:
            fig, ax = plate.plot_analysis_segmentation(
                figsize=figsize
            )
            fig.savefig(f"{dirpath}{plate.sample_name}.png")
            plt.close(fig)

    def save_colony_segmentation(self, dirpath, figsize=(16, 24)):
        for plate in self.plates:
            fig, ax = plate.plot_colony_segmentation(
                figsize=figsize
            )
            fig.savefig(f"{dirpath}{plate.sample_name}.png")
            plt.close(fig)

    def save_plate_gridding_op(self, dirpath, figsize=(12, 8)):
        for plate in self.plates:
            fig, ax = plate.plot_well_grid(figsize=figsize)
            ax.set_title(f"{plate.sample_name}")
            fig.savefig(f"{dirpath}{plate.sample_name}.png")
            plt.close(fig)

