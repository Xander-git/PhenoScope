import matplotlib.pyplot as plt
import numpy.ma as ma
import numpy as np

from ._colony_profile_object import ColonyProfileObject


class ColonyProfilePlotObject(ColonyProfileObject):
    def plotAx_input(self, ax, use_grayscale=False, cmap="viridis"):
        if use_grayscale:
            img = self.gray_img
        else:
            img = self.input_img
        with plt.ioff():
            if use_grayscale:
                ax.imshow(img, cmap=cmap)
            else:
                ax.imshow(img)
            ax.set_title("Input Image")

    def plotAx_colony(self, ax, use_grayscale=False, cmap="viridis"):
        if self.colony_mask is None:
            self.status_validity = False
            if use_grayscale:
                img = self.gray_img
            else:
                img = self.input_img
        else:
            if use_grayscale:
                img = self.masked_gray_img.filled(0)
            else:
                img = self.masked_img.filled(0)

        with plt.ioff():
            if use_grayscale:
                ax.imshow(img, cmap=cmap)
            else:
                ax.imshow(img)

            if self.status_validity is False:
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.spines['bottom'].set(color="red", linewidth=5)
                ax.spines['top'].set(color="red", linewidth=5)
                ax.spines['right'].set(color="red", linewidth=5)
                ax.spines['left'].set(color="red", linewidth=5)
            else:
                ax.set_axis_off()
            ax.set_title("Colony")

    def plotAx_unfiltered(self, ax, cmap="viridis"):
        with plt.ioff():
            if self.status_validity:
                # ax.imshow(self.unfiltered_segmentation, cmap=cmap)
                ax.imshow(
                    self.labeled_segmentation,
                    cmap=cmap
                )
                ax.set_title("Segmented Colony")
            else:
                ax.imshow(self.gray_img, cmap="YlOrRd")
                ax.set_title(f"Err:{self.image_name}")
        return ax

    def plotAx_segmentation(self, ax, cmap="viridis"):
        with plt.ioff():
            if self.status_validity:
                # ax.imshow(self.segmentation, cmap=cmap)
                ax.imshow(
                    self.colony_mask,
                    cmap=cmap
                )
                ax.set_title("Segmented Colony")
            else:
                ax.imshow(self.gray_img, cmap="YlOrRd")
                ax.set_title("Invalid Segmentation")
        return ax

    def plot_colony(self, use_grayscale=False, cmap="viridis"):
        with plt.ioff():
            fig, ax = plt.subplots()
            self.plotAx_colony(ax=ax, use_grayscale=use_grayscale, cmap=cmap)
        return fig, ax

    def plot_unfiltered(self, cmap='viridis'):
        with plt.ioff():
            fig, ax = plt.subplots()
            self.plotAx_unfiltered(ax, cmap)
        return fig, ax

    def plot_segmentation(self, cmap='viridis'):
        with plt.ioff():
            fig, ax = plt.subplots()
            self.plotAx_segmentation(ax, cmap)
        return fig, ax
