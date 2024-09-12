# ----- Logger ------
import logging

formatter = logging.Formatter(
        fmt=f'[%(asctime)s|%(name)s] %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S'
)
console_handler = logging.StreamHandler()
log = logging.getLogger(__name__)
log.addHandler(console_handler)
console_handler.setFormatter(formatter)
# ----- Imports -----

# ----- Pkg Relative Import -----
from ._plate_base import PlateBase
from ..detection.blob_finder import BlobFinder
from ..util.plotting import plot_plate_rows


# ----- Main Class Definition -----
class PlateBlobs(PlateBase):
    def __init__(self, img, n_rows=8, n_cols=12, **kwargs):
        super().__init__(
                img=img,
                n_rows=n_rows,
                n_cols=n_cols
        )

        self.blob_finder_settings = {
            'detection_method': kwargs.get("detection_method", "log"),
            'min_sigma': kwargs.get("min_sigma", 4),
            'max_sigma': kwargs.get("max_sigma", 40),
            'num_sigma': kwargs.get("num_sigma", 45),
            'threshold': kwargs.get("threshold", 0.01),
            'overlap': kwargs.get("overlap", 0.1),
            'min_size': kwargs.get("min_size", 180),
            'filter_threshold_method': kwargs.get("filter_threshold_method", "triangle"),
            'tophat_radius': kwargs.get("tophat_radius", 15),
            'border_filter': kwargs.get("border_filter", 50),
        }

        self.blobs = BlobFinder(
                n_rows=self.n_rows,
                n_cols=self.n_cols,
                blob_search_method=self.blob_finder_settings['detection_method'],
                min_sigma=self.blob_finder_settings['min_sigma'],
                max_sigma=self.blob_finder_settings['max_sigma'],
                num_sigma=self.blob_finder_settings['num_sigma'],
                threshold=self.blob_finder_settings['threshold'],
                max_overlap=self.blob_finder_settings['overlap'],
                min_size=self.blob_finder_settings['min_size'],
                filter_threshold_method=self.blob_finder_settings['filter_threshold_method'],
                tophat_radius=self.blob_finder_settings['tophat_radius'],
                border_filter=self.blob_finder_settings['border_filter'],
        )

        self.status_initial_blobs = False

        self._invalid_op = "Normalization run without issues"
        self._invalid_op_img = self._invalid_blobs = None

        log.debug("Initialized Class and Set Image")

    def run(self):
        # TODO: Integrate autorun
        super().run()

    def _normalize(self):
        self._update_blobs()

    def _set_op(self, op_name, op_img, op_blobs):
        self._invalid_op = op_name
        self._invalid_op_img = op_img
        self._invalid_blobs = op_blobs

    def _update_blobs(self):
        self.blobs.find_blobs(self.img)
        if self.status_initial_blobs is False:
            log.info("Performing initial blob search")
            self.status_initial_blobs = True

    def _plotAx_failed_normalization(self, ax):
        if self._invalid_blobs is not None:
            plot_plate_rows(img=self._invalid_op_img,
                            blobs_class=self._invalid_blobs,
                            ax=ax)
            ax.set_title(self._invalid_op)
        else:
            ax.imshow(self._invalid_op_img)
            ax.set_title(self._invalid_op)
