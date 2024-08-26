# ----- Imports -----
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.util import img_as_ubyte

import logging

formatter = logging.Formatter(fmt=f'[%(asctime)s|%(name)s] %(levelname)s - %(message)s',
                              datefmt='%m/%d/%Y %I:%M:%S')
console_handler = logging.StreamHandler()
log = logging.getLogger(__name__)
log.addHandler(console_handler)
console_handler.setFormatter(formatter)

# ----- Pkg Relative Import -----
from ._plate_grid import PlateGrid


# ----- Main Class Definition -----
class PlateIO(PlateGrid):
    def __init__(self, img: np.ndarray,
                 n_rows: int = 8,
                 n_cols: int = 12,
                 border_padding: int = 50,
                 **kwargs
                 ):
        super().__init__(
                img=img,
                n_rows=n_rows,
                n_cols=n_cols,
                border_padding=border_padding,
                **kwargs
        )

    def imsave(self, fname_save):
        if self.status_validity:
            log.info(f"Saving normalized plate img to {fname_save}")
            io.imsave(fname_save, img_as_ubyte(self.img), check_contrast=False, quality=100)
        else:
            fig, ax = self._plot_invalid()
            fname = fname_save.split(".")
            fname_save = fname[0] + "_invalid_normalization" + fname[1]
            fig.savefig(fname_save)

    def save_ops(self, fname_save):
        '''
        Saves operation figures to fname_save. This function changes depending on the amount of operations from the start to the endpoint
        '''
        log.info(f"Saving plate normalization operations to {fname_save}")
        if self.status_validity:
            fig, ax = self.plot_ops()
            fig.savefig(fname_save)
        else:
            fig, ax = self._plot_invalid()
            fname_split = fname_save.split(".")
            fig.savefig(fname_split[0] + "_invalid_normalizaton" + fname_split[1])
        plt.close(fig)

    def plot_ops(self, plate_name=None, figsize=(18, 14), tight_layout=True, fontsize=16):
        with plt.ioff():
            fig, ax = plt.subplots(
                    nrows=2, ncols=2,
                    figsize=figsize, tight_layout=tight_layout
            )
            opAx = ax.ravel()
            self.plotAx_alignment(ax=opAx[0], fontsize=fontsize)
            self.plotAx_fitting(ax=opAx[1], fontsize=fontsize)
            self.plotAx_well_grid(ax=opAx[2], fontsize=fontsize)
            opAx[3].imshow(self.img)
            opAx[3].set_title("Final Image", fontsize=fontsize)
            opAx[3].grid(False)
            if plate_name is not None:
                fig.suptitle(plate_name, fontsize=fontsize)
        return fig, ax

    def save_wells(self, dirpath_folder, name_prepend="", filetype=".png"):
        if dirpath_folder[-1] == "/": dirpath_folder = dirpath_folder[:-1]
        well_imgs = self.get_well_imgs()
        log.info(f"Saving well images to {dirpath_folder}")
        for idx, well in enumerate(well_imgs):
            try:
                io.imsave(
                        fname=f"{dirpath_folder}/{name_prepend}well({idx:03d}){filetype}",
                        arr=img_as_ubyte(well),
                        quality=100,
                        check_contrast=False
                )
            except:
                log.warning(f"Could not save well {idx}", exc_info=True)

    def _plot_invalid(self):
        with plt.ioff():
            fig, ax = plt.subplots()
            self._plotAx_failed_normalization(ax)
        return fig, ax
