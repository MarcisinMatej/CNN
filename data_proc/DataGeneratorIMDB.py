from data_proc.DataGeneratorOnLine import DataGeneratorOnLine
from data_proc.DataLoaderCelebA import load_attr_vals_txts

LABEL_DICT_PATH = "data_proc/encoded_labels.npy"
IMAGES_FOLDER = "data_proc/CelebA/img_align_celeba/"


def load_config_IMDB():
    return [],[]


class DataGeneratorIMDB(DataGeneratorOnLine):
    def __init__(self, img_shape=(100, 100), chunk_size=1024):
        """

        :param img_shape: resolution of final image
        :param chunk_size: size of super batch
        :param rot_int: interval for image rotation
        """
        'Initialization'
        self.img_shape = img_shape
        self.chunk_size = chunk_size
        self.attr_vals = load_attr_vals_txts()
        self.attr_map, self.coord_dict = load_config_IMDB()
        # count how many different attributes we will predict
        self.attr_cnt = len(self.attr_vals)
        # split data to training,testing,validation
        self.train_ids = []
        self.test_ids = []
        self.validation_ids = []
        self.find_split_ids_IMDB()

    def find_split_ids_IMDB(self):
        pass