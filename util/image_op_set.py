import matplotlib.pyplot as plt
from .image_op import ImageOp

class ImageOpSet:
    op_set = []

    def add_op(self, name, image, dataset_name=None):
        self.op_set.append(ImageOp(name, image, dataset_name))
