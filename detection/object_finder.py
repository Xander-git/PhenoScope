import numpy as np

from ._object_finder_view import ObjectFinderView

"""
End-point class for ObjectFinder
"""


class ObjectFinder(ObjectFinderView):
    def find_objects(self, image: np.ndarray) -> None:
        super().find_objects(image=image)
        return self.results

    pass
