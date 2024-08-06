# ----- Imports -----
import math
from skimage.filters import threshold_otsu, threshold_triangle
from skimage.morphology import white_tophat, disk
from scipy.ndimage import binary_fill_holes
# ----- Pkg Relative Import -----
from .blob_finder_table import BlobFinderTable


# ----- Main Class Definition -----
# TODO: Add more filters?
class BlobFinderParticleFilter(BlobFinderTable):
    '''
    Last Updated: 7/8/2024
    '''
    def __init__(self, gray_img, n_rows=8, n_cols=12, blob_search_method="log",
                 min_sigma=4, max_sigma=40, num_sigma=45,
                 threshold=0.01, max_overlap=0.1,
                 min_size: int = 180,
                 filter_threshold_method="triangle",
                 tophat_radius=15,
                 border_filter: int = 50
                 ):
        if filter_threshold_method is None:
            pass
        else:
            gray_img = self._filter_by_threshold(gray_img, filter_threshold_method)
            gray_img = self._filter_by_white_tophat(gray_img, tophat_radius)
        super().__init__(gray_img, n_rows, n_cols, blob_search_method,
                         min_sigma, max_sigma, num_sigma,
                         threshold, max_overlap
                         )
        self._filter_by_size(min_size)
        self._filter_by_border(gray_img.shape, border_filter)
        self.generate_table()

    def _find_circle_info(self):
        self._table['radius'] = self._table['sigma'] * math.sqrt(2)
        self._table.drop(columns='sigma')
        self._table['area'] = math.pi * (self._table['radius'] * self._table['radius'])
        self._table["radius"] = self._table["radius"] + self._table["radius"]*0.05

        if "id" not in self._table.columns:
            self._table = self._table.reset_index(drop=False).rename(columns={
                "index": "id"
            })
        self._table = self._table.loc[:, ["id", 'x', 'y', 'sigma', 'radius', 'area']].reset_index(drop=True)

    @staticmethod
    def _filter_by_threshold(gray_img, threshold_method):
        if threshold_method is None:
            return gray_img
        elif threshold_method == "triangle":
            thresh = threshold_triangle(gray_img)
        elif threshold_method == "otsu":
            thresh = threshold_otsu(gray_img)
        else:
            thresh = threshold_triangle(gray_img)
        gray_img = gray_img > thresh
        gray_img = binary_fill_holes(gray_img)
        return gray_img

    @staticmethod
    def _filter_by_white_tophat(gray_img, radius):
        footprint = disk(radius)
        res = white_tophat(gray_img, footprint)
        gray_img = gray_img & ~res
        return gray_img

    def _filter_by_size(self, min_size):
        self._table = self._table[self._table['area'] >= min_size]
        self._table = self._table.reset_index(drop=True)

    def _filter_by_border(self, img_shape, border_filter):
        self._table = self._table[self._table['y_minus'] > 0 + border_filter]
        self._table = self._table[self._table['x_minus'] > 0 + border_filter]
        self._table = self._table[self._table['y_plus'] < img_shape[0] - border_filter]
        self._table = self._table[self._table['x_plus'] < img_shape[1] - border_filter]
