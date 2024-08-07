import matplotlib.pyplot as plt

from ._plate_profile_base import PlateProfileBase

import logging
log = logging.getLogger(__file__)
logging.basicConfig(format=f'[%(asctime)s|%(levelname)s|%(name)s] %(message)s')


class PlateProfileAnalysis(PlateProfileBase):
    def plot_analysis(self, fontsize=18):
        self.validate_well_names()
        with plt.ioff():
            fig, axes = plt.subplots(nrows=self.n_rows, ncols=self.n_cols, figsize=(16, 12),
                                     sharex=True, sharey=True)
            for idx, ax in enumerate(axes.ravel()):
                self.wells[idx].plotAx_segmentation(ax)
                # if self.well_validity[idx]:
                #     ax.imshow(self.segmentation_imgs[idx], cmap="viridis")
                # else:
                #     ax.imshow(self.gray_imgs[idx], cmap="gray")
                ax.set_axis_off()
                ax.set_title(f"{self.wells[idx].well_name}")
            fig.suptitle(self.sample_name, fontsize=fontsize)
        return fig, axes

    def plot_unfiltered(self, fontsize=18):
        self.validate_well_names()
        with plt.ioff():
            fig, axes = plt.subplots(nrows=self.n_rows, ncols=self.n_cols, figsize=(16, 12),
                                     sharex=True, sharey=True, tight_layout=True)
            for idx, ax in enumerate(axes.ravel()):
                self.wells[idx].plotAx_unfiltered(ax)
                # if self.well_validity[idx]:
                #     ax.imshow(self.unfiltered_imgs[idx], cmap="viridis")
                # else:
                #     ax.imshow(self.gray_imgs[idx], cmap="gray")
                ax.set_axis_off()
                ax.set_title(f"{self.wells[idx].well_name}")
            fig.suptitle(self.sample_name, fontsize=fontsize)
        return fig, axes

    def plot_analysis_segmentation(self, figsize=(18, 26), fontsize_title=26, fontsize_subtitle=16):
        with plt.ioff():
            fig, axes = plt.subplots(
                nrows=(self.n_rows * 2)+1,
                ncols=self.n_cols,
                figsize=figsize,
                sharey=True,
                sharex=True,
                tight_layout=True
            )
            ax_ravel = axes.ravel()
            count = 0
            self.validate_well_names()
            for idx in range(len(self.wells)):
                ax = ax_ravel[count]
                self.wells[idx].plotAx_unfiltered(ax)
                ax.set_axis_off()
                ax.set_title(f"IdObj: {idx}", fontsize=fontsize_subtitle)
                count += 1
                self.validate_well_names()
            for idx in range(self.n_cols):
                ax = ax_ravel[count]
                ax.axis("off")
                count += 1
            for idx in range(len(self.wells)):
                ax = ax_ravel[count]
                self.wells[idx].plotAx_segmentation(ax)
                ax.set_axis_off()
                ax.set_title(f"Filter: {idx}", fontsize=fontsize_subtitle)
                count += 1
            fig.suptitle(f"{self.sample_name}", fontsize=fontsize_title)
        return fig, ax
