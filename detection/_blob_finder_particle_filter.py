# ----- Imports -----
import math
from skimage.filters import threshold_otsu, threshold_triangle
from skimage.morphology import white_tophat, disk, square
from scipy.ndimage import binary_fill_holes
# ----- Pkg Relative Import -----
from ._blob_finder_table import BlobFinderTable


# ----- Main Class Definition -----
# TODO: Add more filters?
class BlobFinderParticleFilter(BlobFinderTable):
    def __init__(self, gray_img, n_rows: int = 8, n_cols: str = 12, blob_search_method: str = "log",
                 min_sigma: int = 4, max_sigma: int = 40, num_sigma: int = 45,
                 threshold: float = 0.01, max_overlap: float = 0.1,
                 min_size: int = 180,
                 filter_threshold_method: str = "triangle",
                 tophat_shape: str = "square",
                 tophat_radius: int = 12,
                 border_filter: int = 50
                 ):
        gray_img = self.check_grayscale(gray_img)
        self.filter_threshold_method = filter_threshold_method
        self.tophat_shape = tophat_shape
        self.tophat_radius = tophat_radius

        super().__init__(gray_img=gray_img,
                         n_rows=n_rows,
                         n_cols=n_cols,
                         blob_search_method=blob_search_method,
                         min_sigma=min_sigma,
                         max_sigma=max_sigma,
                         num_sigma=num_sigma,
                         threshold=threshold,
                         max_overlap=max_overlap
                         )
        self._filter_by_size(min_size)
        self._filter_by_border(gray_img.shape, border_filter)
        self.generate_table()

    def search_blobs(self, gray_img, method):
        if self.filter_threshold_method is None:
            pass
        else:
            gray_img = self._filter_by_threshold(gray_img, self.filter_threshold_method)
            gray_img = self._filter_by_white_tophat(gray_img, shape=self.tophat_shape, radius=self.tophat_radius)

        super().search_blobs(gray_img=gray_img, method=method)

    def _find_circle_info(self):
        self._table['radius'] = self._table['sigma'] * math.sqrt(2)
        self._table.drop(columns='sigma')
        self._table['area'] = math.pi * (self._table['radius'] * self._table['radius'])
        # self._table["radius"] = self._table["radius"] + self._table["radius"]*0.05

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
    def _filter_by_white_tophat(gray_img, shape, radius):
        if shape == "disk":
            footprint = disk(radius)
        elif shape == "square":
            footprint = square(radius * 2)
        else:
            footprint = disk(radius)
        res = white_tophat(image=gray_img, footprint=footprint)
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
