
class ImageOp:
    name = None
    image = None
    dataset_name = None
    def __init__(self, name, image, dataset_name = None):
        self.name=name
        self.image = image
        self.dataset_name = dataset_name
        