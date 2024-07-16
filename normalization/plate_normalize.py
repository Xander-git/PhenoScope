# ----- Imports -----
import matplotlib.pyplot as plt

import skimage.io as io
from skimage.util import img_as_ubyte

# ----- Pkg Relative Import -----
from .well_isolation import WellIsolation
from ..util.verbosity import Verbosity

# ----- Main Class Definition -----


class PlateNormalize(WellIsolation): # The parent class will change to the latest endpoint for the pipeline
    '''
    Last Updated: 7/9/2024
    '''
    def __init__(self, img, n_rows=8, n_cols=12,
                 blob_detection_method='log',
                 min_sigma=4, max_sigma=50, num_sigma=30,
                 threshold=0.01, overlap=0.1, min_size=180,
                 border_padding=50, verbose=True, sample_name=None
                 ):
        if sample_name is not None:
            verb = Verbosity(verbose)
            verb.title(sample_name)
        super().__init__(
            img, n_rows, n_cols,
            blob_detection_method,
            min_sigma, max_sigma, num_sigma,
            threshold, overlap, min_size,
            border_padding, verbose
        )

    def imsave(self, fname_save):
        self.verb.start("Saving Plate Image")
        io.imsave(fname_save, img_as_ubyte(self.img), check_contrast=False, quality=100)
        self.verb.end("Saving Plate Image")

    def save_operations(self, fname_save):
        '''
        Saves operation figures to fname_save. This function changes depending on the amound of operations from the start to the endpoint
        '''
        self.verb.start("saving plate operations")
        fig, ax = self.plot_operations()
        fig.savefig(fname_save)
        plt.close(fig)
        self.verb.end("saving plate operations")

    def plot_operations(self, figsize=(18, 14), tight_layout=True, fontsize=18):
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize, tight_layout=tight_layout)
        opAx = ax.ravel()
        with plt.ioff():
            self.plotAx_alignment(opAx[0])
            self.plotAx_fitting(opAx[1])
            self.plotAx_well_grid(opAx[2])
            opAx[3].imshow(self.img)
            opAx[3].set_title("Final Image", fontsize=fontsize)
        return fig, ax

    def save_wells(self, fpath_folder, name_prepend="", filetype=".png"):
        well_imgs = self.get_well_imgs()
        self.verb.start("saving isolated well images")
        for idx, well in enumerate(well_imgs):
            self.verb.body(f"saving well: {idx}")
            io.imsave(
                f"{fpath_folder}/{name_prepend}well_{idx}{filetype}",
                img_as_ubyte(well), quality=100, check_contrast=False
            )
        self.verb.end("saving isolated well images")
