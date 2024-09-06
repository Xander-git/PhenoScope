import logging

formatter = logging.Formatter(
        fmt=f'[%(asctime)s|%(name)s] %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S'
)
console_handler = logging.StreamHandler()
log = logging.getLogger(__name__)
log.addHandler(console_handler)
console_handler.setFormatter(formatter)

import matplotlib.pyplot as plt
from pathlib import Path

from ._plate_series_plotting import PlateSeriesPlotting


class PlateSeriesIO(PlateSeriesPlotting):
    def save_results2csv(self, filepath):
        filepath = Path(filepath)
        if filepath.suffix != ".csv": raise ValueError("filepath must have .csv extension")

        self.results.to_csv(filepath)

    def save_analysis_segmentation(self, dirpath, figsize=(16, 24)):
        dirpath = Path(dirpath)
        for plate in self.plates:
            fig, ax = plate.plot_analysis_segmentation(
                    figsize=figsize
            )
            fig.savefig(dirpath / f"{plate.plate_name}.png")
            plt.close(fig)

    def save_colony_segmentation(self, dirpath, figsize=(16, 12)):
        dirpath = Path(dirpath)
        dirpath.mkdir(exist_ok=True, parents=True)
        for plate in self.plates:
            log.debug(f"plate: {plate}")
            try:
                fig, ax = plate.plot_colony_segmentation(
                        figsize=figsize
                )
                fig.savefig(dirpath / f"{plate.plate_name}.png")
                plt.close(fig)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                log.warning(f"Could not save colony segmentation of plate {plate.plate_name}: {e}", exc_info=True)

    def save_plate_grid_division(self, dirpath, figsize=(12, 8)):
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        for plate in self.plates:
            fig, ax = plate.plot_well_grid(figsize=figsize)
            ax.set_title(f"{plate.plate_name}")
            fig.savefig(dirpath / f"{plate.plate_name}.png")
            plt.close(fig)
