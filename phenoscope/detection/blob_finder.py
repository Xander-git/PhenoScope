from ._blob_finder._blob_finder_bin_mse_filter import BlobFinderBinMSEFilter


class BlobFinder(BlobFinderBinMSEFilter):
    def find_blobs(self, img):
        super().find_blobs(img)
        return self.results
