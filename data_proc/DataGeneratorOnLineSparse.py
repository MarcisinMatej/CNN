from keras.preprocessing import image

from data_proc.DataGeneratorOnLine import DataGeneratorOnLine
from data_proc.DataLoader import load_attr_vals_txts, load_atributes_txts
import numpy as np

from data_proc.ImagePreProcess import load_crop_boxes

LABEL_DICT_PATH = "data_proc/encoded_labels.npy"
IMAGES_FOLDER = "data_proc/CelebA/img_align_celeba/"
CHANCE = 0.25

MASKS = [[True, False, False, False, False],
         [False, True, False, False, False],
         [False, False, True, False, False],
         [False, False, False, True, False],
         [False, False, False, False, True]]


def create_map(attr_vals):
    _map = {}
    for attr_val in attr_vals:
        key = attr_val.split()[0].split("/")[-1]
        values = [i - 1 for i in list(map(int, attr_val.split()[1:]))]
        _map[key] = values
    return _map


class DataGeneratorOnLineSparse(DataGeneratorOnLine):
    """Generates data for Keras"""

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
        self.sparse_attr_map = create_map(load_atributes_txts())
        self.coord_dict = load_crop_boxes()
        # count how many different attributes we will predict
        self.attr_cnt = len(self.attr_vals)
        self.attr_class_cnt = []
        # count how many classes are in each label (length of one-hot true value vector)
        for attr_val in self.attr_vals:
            cnt = len(attr_val.split(":")[1].split(","))
            self.attr_class_cnt.append(cnt)
        # initialize and split data to training,testing,validation
        self.train_ids = []
        self.test_ids = []
        self.validation_ids = []
        self.find_split_ids()

    def get_encoded_labels(self, keys):
        """
        Generate labels from attribute file for list of keys,
        the labels are returned in the same order as corresponding
        keys in parameter list.
        :param keys: list of labels in string format
        :return: labels for specific batch of data in one-hot encoded format
        """
        to_return = []
        for key in keys:
            to_return.append(self.sparse_attr_map[key])
        # need to transform to N arrays, as KERAS requires all labels for one output/attribute
        # in single array, so for 5 attributes and bulk 1024, it will be 5 arrays of length
        # 10240
        return [np.array(tmp_arr) for tmp_arr in zip(*to_return)]

    def get_encoded_labels_h(self, keys, mask):
        """
        Generate labels from attribute file for list of keys,
        the labels are returned in the same order as corresponding
        keys in parameter list.
        :param keys: list of labels in string format
        :return: labels for specific batch of data in one-hot encoded format
        """
        to_return = []
        for key in keys:
            to_return.append(self.hide_values(self.sparse_attr_map[key], mask))
        # need to transform to N arrays, as KERAS requires all labels for one output/attribute
        # in single array, so for 5 attributes and bulk 1024, it will be 5 arrays of length
        # 10240
        return [np.array(tmp_arr) for tmp_arr in zip(*to_return)]

    def hide_values(self, vals, mask):
        to_ret = []
        for val, m in zip(vals, mask):
            if m:
                to_ret.append(val)
            else:
                to_ret.append(-1)
        return to_ret


