# ----- Imports -----
from skimage.filters import threshold_otsu, threshold_triangle
from skimage.morphology import white_tophat, disk
# ----- Pkg Relative Import -----
from .blob_finder_table import BlobFinderTable


# ----- Main Class Definition -----
# TODO: Add more filters?
class BlobFinder(BlobFinderTable):
    '''
    Last Updated: 7/8/2024
    '''

    def __init__(self, gray_img, n_rows=8, n_cols=12, blob_detect_method="log",
                 min_sigma=4, max_sigma=40, num_sigma=45,
                 threshold=0.01, overlap=0.1,
                 min_size: int = 180, filter_threshold_method="triangle", tophat_radius=10,
                 border_filter: int = 50
                 ):
        if filter_threshold_method is None:
            pass
        else:
            gray_img = self._filter_by_threshold(gray_img, filter_threshold_method)
            gray_img = self._filter_by_white_tophat(gray_img, tophat_radius)
        super().__init__(gray_img, n_rows, n_cols, blob_detect_method,
                         min_sigma, max_sigma, num_sigma,
                         threshold, overlap
                         )
        self._filter_by_size(min_size)
        self._filter_by_border(gray_img.shape, border_filter)
        self.generate_table()

    def _filter_by_threshold(self, gray_img, threshold_method):
        if threshold_method == "triangle":
            thresh = threshold_triangle(gray_img)
        elif threshold_method == "otsu":
            thresh = threshold_otsu(gray_img)
        else:
            thresh = threshold_triangle(gray_img)
        gray_img = gray_img > thresh
        return gray_img

    def _filter_by_white_tophat(self, gray_img, radius):
        footprint = disk(radius)
        res = white_tophat(gray_img, footprint)
        gray_img = gray_img & ~res
        return gray_img

    def _filter_by_size(self, min_size):
        self.table = self.table[self.table['area'] >= min_size]
        self.table = self.table.reset_index(drop=True)

    def _filter_by_border(self, img_shape, border_filter):
        self.table = self.table[self.table['y_minus'] > 0 + border_filter]
        self.table = self.table[self.table['x_minus'] > 0 + border_filter]
        self.table = self.table[self.table['y_plus'] < img_shape[0] - border_filter]
        self.table = self.table[self.table['x_plus'] < img_shape[1] - border_filter]
