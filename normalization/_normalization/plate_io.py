# ----- Imports -----
import matplotlib.pyplot as plt

import skimage.io as io
from skimage.util import img_as_ubyte

import os
import logging
logger_name = "phenomics-normalization"
log = logging.getLogger(logger_name)
logging.basicConfig(format=f'[%(asctime)s|%(levelname)s|{os.path.basename(__file__)}] %(message)s')

# ----- Pkg Relative Import -----
from .plate_grid import PlateGrid

# ----- Main Class Definition -----


class PlateIO(PlateGrid):
    '''
    Last Updated: 7/9/2024
    '''
    def __init__(self, img, n_rows=8, n_cols=12,
                 align=True, fit=True,
                 auto_run=True
                 ):
        super().__init__(
            img, n_rows, n_cols,
            align, fit
        )
        if auto_run:
            self.run()

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
        Saves operation figures to fname_save. This function changes depending on the amound of operations from the start to the endpoint
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

    def plot_ops(self, figsize=(18, 14), tight_layout=True, fontsize=24, plate_name = None):
        with plt.ioff():
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize, tight_layout=tight_layout)
            opAx = ax.ravel()
            self.plotAx_alignment(opAx[0], fontsize)
            self.plotAx_fitting(opAx[1], fontsize)
            self.plotAx_well_grid(opAx[2], fontsize)
            opAx[3].imshow(self.img)
            opAx[3].set_title("Final Image", fontsize=fontsize)
            opAx[3].grid(False)
            if plate_name is not None:
                fig.suptitle(plate_name, fontsize=fontsize)
        return fig, ax

    def save_wells(self, dirpath_folder, name_prepend="", filetype=".png"):
        well_imgs = self.get_well_imgs()
        log.info(f"Saving well images to {dirpath_folder}")
        for idx, well in enumerate(well_imgs):
            try:
                io.imsave(
                    f"{dirpath_folder}{name_prepend}well({idx:03d}){filetype}",
                    img_as_ubyte(well), quality=100, check_contrast=False
                )
            except:
                log.warning(f"Could not save well {idx}")
                self.invalid_wells.append(idx)

    def _plot_invalid(self):
        with plt.ioff():
            fig, ax = plt.subplots()
            self._plotAx_failed_normalization(ax)
        return fig, ax
