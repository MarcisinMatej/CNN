from data_proc.ConfigLoaderOUI import load_config_imdb
from data_proc.DataGeneratorCelebA import DataGeneratorCelebA
from data_proc.ConfigLoaderCelebA import load_attr_vals_txts
from keras.preprocessing import image
import numpy as np

from data_proc.ImageHandler import invalid_img, get_image

# IMAGES_FOLDER_IMDB = "/datagrid/personal/marcisin/"
IMAGES_FOLDER_IMDB = "data_proc/data/adience/"

class DataGeneratorOUI(DataGeneratorCelebA):
    def __init__(self, img_shape=(100, 100), chunk_size=12400):
        """
        Initialization for generator of Adience database.
        :param img_shape: resolution of final image
        :param chunk_size: size of super batch
        :param rot_int: interval for image rotation
        """
        'Initialization'
        self.img_source = IMAGES_FOLDER_IMDB
        self.img_shape = img_shape
        self.chunk_size = chunk_size
        self.attr_vals = load_attr_vals_txts()
        # count how many different attributes we will predict
        self.attr_cnt = len(self.attr_vals)
        # split data to training,testing,validation
        self.train_ids, self.validation_ids, self.test_ids, self.attr_map, self.coord_dict = load_config_imdb(True)
        # load images
        self.training_data, err_t = self.get_images_online(self.train_ids)
        self.validation_data, err_v = self.get_images_online(self.validation_ids)
        # update labels accordingly to loaded images
        self.train_ids = self.get_encoded_labels([name for name in self.train_ids if name not in err_t])
        self.validation_ids = self.get_encoded_labels([name for name in self.validation_ids if name not in err_v])
        print("-- Generator OUI Adience initialized.")

    def get_encoded_labels(self, keys):
        """
        Generate labels from attribute file for list of keys,
        the labels are returned in the same order as corresponding
        keys in parameter list.
        :param keys: list of labels in string format
        :return: labels for specific batch of data raw form (0-n)
        """
        to_return = []
        for key in keys:
            to_return.append(self.attr_map[key])
        # need to transform to N arrays, as KERAS requires all labels for one output/attribute
        # in single array, so for 5 attributes and bulk 1024, it will be 5 arrays of length
        # 10240
        return [np.array(tmp_arr) for tmp_arr in zip(*to_return)]

    def generate_training(self):
        yield self.training_data, self.train_ids

    def generate_validation(self):
        yield self.validation_data, self.validation_ids

