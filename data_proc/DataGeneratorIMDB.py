from data_proc.ConfigLoaderIMDB import load_config_imdb
from data_proc.DataGeneratorCelebA import DataGeneratorCelebA
from data_proc.ConfigLoaderCelebA import load_attr_vals_txts
from keras.preprocessing import image
import numpy as np

from data_proc.ImageHandler import invalid_img, get_image

# IMAGES_FOLDER_IMDB = "/datagrid/personal/marcisin/"
IMAGES_FOLDER_IMDB = "data_proc/data/imdb/"


class DataGeneratorIMDB(DataGeneratorCelebA):
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
        # count how many different attributes we will predict
        self.attr_cnt = len(self.attr_vals)
        # split data to training,testing,validation
        self.train_ids, self.validation_ids, self.test_ids, self.attr_map, self.coord_dict = load_config_imdb()
        self.img_source = IMAGES_FOLDER_IMDB

        self.training_data, err_t = self.get_images_online(self.train_ids)
        self.validation_data, err_v = self.get_images_online(self.validation_ids)

        self.train_ids = self.get_encoded_labels([name for name in self.train_ids if name not in err_t])
        self.validation_ids = self.get_encoded_labels([name for name in self.validation_ids if name not in err_v])

        print("-- Generator IMDB initialized.")

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
            to_return.append(self.attr_map[key])
        # need to transform to N arrays, as KERAS requires all labels for one output/attribute
        # in single array, so for 5 attributes and bulk 1024, it will be 5 arrays of length
        # 10240
        return [np.array(tmp_arr) for tmp_arr in zip(*to_return)]

    def get_images_online(self, img_names):
        """
        Reads list of images from specidied folder.
        The images are resized to self.img_shape specified
        in the generator contructor.
        In case of error, image is not added to return list
        and error is just printed.
        :param img_names: List of image names
        :return: list of vstacked images, channel_last format
        """
        images = []
        errs = []
        for img_name in img_names:
            try:
                path = self.img_source + img_name
                # print(path)
                img = get_image(path, self.img_shape)
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                images.append(x)
            except Exception as e:
                # print(str(e))
                errs.append(img_name)

        print("-- Failed to load ", str(len(errs)), " images.")
        return np.vstack(images), errs

    def generate_training(self):
        yield self.training_data, self.train_ids

    def generate_validation(self):
        yield self.validation_data, self.validation_ids
