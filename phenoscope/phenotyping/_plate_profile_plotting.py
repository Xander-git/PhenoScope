import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from skimage.io import imsave
from skimage.util import img_as_ubyte

from ._plate_profile_base import PlateProfileBase

import logging

log = logging.getLogger(__file__)
logging.basicConfig(format=f'[%(asctime)s|%(levelname)s|%(name)s] %(message)s')


class PlateProfilePlotting(PlateProfileBase):
    def plot_analysis(self, fontsize=18):
        # self.validate_well_names()
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
                ax.set_title(f"{self.wells[idx].plate_name}")
            fig.suptitle(self.plate_name, fontsize=fontsize)
        return fig, axes

    def plot_unfiltered(self, fontsize=18):
        # self.validate_well_names()
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
                ax.set_title(f"well({idx})")
            fig.suptitle(self.plate_name, fontsize=fontsize)
        return fig, axes

    def plot_analysis_segmentation(self, figsize=(18, 26), fontsize_title=26, fontsize_subtitle=16):
        with plt.ioff():
            fig, axes = plt.subplots(
                    nrows=(self.n_rows * 2) + 1,
                    ncols=self.n_cols,
                    figsize=figsize,
                    sharey=True,
                    sharex=True,
                    tight_layout=False
            )
            ax_ravel = axes.ravel()
            count = 0
            # self.validate_well_names()
            for idx in range(len(self.wells)):
                ax = ax_ravel[count]
                self.wells[idx].plotAx_unfiltered(ax)
                ax.set_axis_off()
                ax.set_title(f"IdObj: {idx}", fontsize=fontsize_subtitle)
                count += 1
                # self.validate_well_names()
            for idx in range(self.n_cols):
                ax = ax_ravel[count]
                if idx == 0:
                    msg = f"Plate Name: {self.plate_name}"
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    mid_y = ylim[0] + (ylim[1] - ylim[0]) / 2
                    ax.text(x=xlim[0], y=mid_y, s=f"{msg}", fontsize=fontsize_title, weight="bold")
                ax.set_axis_off()
                count += 1
            for idx in range(len(self.wells)):
                ax = ax_ravel[count]
                self.wells[idx].plotAx_segmentation(ax)
                ax.set_axis_off()
                ax.set_title(f"Filter: {idx}", fontsize=fontsize_subtitle)
                count += 1
        return fig, ax

    def plot_colony_segmentation(
            self, figsize=(18, 8),
            buffer_width=5,
            fontsize_title=26, fontsize_subtitle=14
    ):
        if fontsize_subtitle > 2:
            fontsize_pass = fontsize_subtitle - 2
        else:
            fontsize_pass = fontsize_subtitle
        with plt.ioff():
            fig, axes = plt.subplots(
                    nrows=self.n_rows,
                    ncols=self.n_cols,
                    figsize=figsize,
                    sharey=True,
                    sharex=True,
                    tight_layout=True
            )
            fig.suptitle(f"Plate Name: {self.plate_name}", fontsize=fontsize_title)
            # self.validate_well_names()
            for idx, ax in enumerate(axes.ravel()):
                img = self.wells[idx].input_img
                buffer = np.full((img.shape[0], buffer_width, 3), 255)
                if self.wells[idx].status_validity:
                    ax.set_title(f"Pass: {idx}", fontsize=fontsize_pass)
                    colony_img = self.wells[idx].masked_img.filled(0)
                    ax.set_axis_off()

                else:
                    ax.set_title(f"Invalid: {idx}", fontsize=fontsize_subtitle, weight='bold')
                    colony_img = self.wells[idx].input_img
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    ax.spines['bottom'].set(color="red", linewidth=5)
                    ax.spines['top'].set(color="red", linewidth=5)
                    ax.spines['right'].set(color="red", linewidth=5)
                    ax.spines['left'].set(color="red", linewidth=5)
                side_by_side = np.concatenate([img, buffer, colony_img], axis=1)
                ax.imshow(side_by_side)
        return fig, ax

    def get_colony_segmentation_operation(self, buffer_width: int = 10):
        side_by_side_set = []
        for well in self.wells:
            _img = well.input_img
            buffer = np.full((_img.shape[0], buffer_width, 3), fill_value=255)
            if well.status_validity:
                colony_img = well.masked_img.filled(0)
            else:
                colony_img = well.input_img
            side_by_side_set.append(np.concatenate([_img, buffer, colony_img], axis=1))
        return side_by_side_set

    def save_colony_segmentation_operation(self, dirpath, image_type=".png", buffer_width: int = 10):
        savepath = Path(dirpath)
        savepath.mkdir(parents=True, exist_ok=True)
        segmentation_operation_imgs = self.get_colony_segmentation_operation(buffer_width=buffer_width)
        for idx, img in enumerate(segmentation_operation_imgs):
            filename = self.plate_name + "+" + self.wells[idx].colony_name + image_type
            imsave(fname=savepath / filename, arr=img_as_ubyte(img), check_contrast=False)
