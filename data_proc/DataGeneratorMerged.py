from data_proc.DataGeneratorOnLine import DataGeneratorOnLine
from data_proc.DataGeneratorOnLineSparse import create_map
from data_proc.DataGeneratorWiki import load_config_wiki
from data_proc.DataLoaderCelebA import load_attr_vals_txts, load_atributes_txts
import numpy as np
from keras.preprocessing import image

from data_proc.ImageParser import get_image
from data_proc.ImagePreProcess import load_crop_boxes

IMAGES_FOLDER_WIKI = "data_proc/data/"
IMAGES_FOLDER_CELEB = "data_proc/data/celebA/"
CONF_FILE_WIKI = "wiki_cat_merged.txt"


def load_config_merged(conf_file):
    train = []
    val = []
    test = []
    tmp = {}
    with open("data_proc/config_files/"+conf_file) as f:
        lines = f.readlines()
        for line in lines:
            tmp[line.split(",")[0]] = [int(i) for i in line.split(",")[1:7]]
            if line.split(",")[-1] == "0\n":
                train.append(line.split(",")[0])
            if line.split(",")[-1] == "1\n":
                val.append(line.split(",")[0])
            if line.split(",")[-1] == "2\n":
                test.append(line.split(",")[0])
    return train, val, test, tmp

class DataGeneratorMerged(DataGeneratorOnLine):
    def __init__(self, img_shape=(100, 100), chunk_size=1024):
        """

        :param img_shape: resolution of final image
        :param chunk_size: size of super batch
        :param rot_int: interval for image rotation
        """
        'Initialization'
        self.img_shape = img_shape
        self.chunk_size = chunk_size
        self.attr_map_celeb = create_map(load_atributes_txts())
        self.coord_dict = load_crop_boxes()
        # split data to training,testing,validation
        self.train_ids_w, self.validation_ids_w, self.test_ids_w, self.attr_map_w = load_config_merged(CONF_FILE_WIKI)

        self.train_ids = []
        self.test_ids = []
        self.validation_ids = []
        self.find_split_ids()

    def generate_data_m(self, pict_ids_w, pict_ids_c):
        """
        Generates data with hiding attributes according to MASKs
        :param pict_ids: ids of pictures
        :return:
        """
        # Generate Wiki data
        indx = 0
        to = indx + self.chunk_size
        while indx <= len(pict_ids_w):
            images, errs = self.get_images_online_w(pict_ids_w[indx: to])
            if len(errs) > 0:
                # get only labels for images which were correctly loade
                img_labels = self.get_raw_labs_w(
                    [name for name in pict_ids_w[indx: to] if name not in errs])
            else:
                img_labels = self.get_raw_labs_w(pict_ids_w[indx: to])
            # get next boundaries
            to += self.chunk_size
            indx += self.chunk_size
            if to != len(pict_ids_w) and (indx + self.chunk_size) > len(pict_ids_w):
                # chunk increase overflow, we need to get the last chunk of data, which is smaller then defined
                to = len(pict_ids_w)

            yield images, img_labels

        # Generate Celeb Data
        indx = 0
        to = indx + self.chunk_size
        while indx <= len(pict_ids_c):
            images, errs = self.get_images_online(pict_ids_c[indx: to],)
            if len(errs) > 0:
                # get only labels for images which were correctly loade
                img_labels = self.get_raw_labs_c([name for name in pict_ids_c[indx: to] if name not in errs])
            else:
                img_labels = self.get_raw_labs_c(pict_ids_c[indx: to])
            # get next boundaries
            to += self.chunk_size
            indx += self.chunk_size
            if to != len(pict_ids_c) and (indx + self.chunk_size) > len(pict_ids_c):
                # chunk increase overflow, we need to get the last chunk of data, which is smaller then defined
                to = len(pict_ids_c)

            yield images, img_labels

    def get_raw_labs_c(self, keys):
        """
       Generate raw labels from attribute file for list of keys.
       :param keys: list of labels in string format
       :return: labels for specific batch of data in one-hot encoded format
       """
        to_return = []
        for key in keys:
            to_return.append(self.attr_map_celeb[key])
            # add missing label for age
            to_return[-1].append(-1)
        # need to transform to N arrays, as KERAS requires all labels for one output/attribute
        # in single array, so for 5 attributes and bulk 1024, it will be 5 arrays of length
        # 10240
        return [np.array(tmp_arr) for tmp_arr in zip(*to_return)]

    def get_raw_labs_w(self, keys):
        """
       Generate raw labels from attribute file for list of keys.
       :param keys: list of labels in string format
       :return: labels for specific batch of data in one-hot encoded format
       """
        to_return = []
        for key in keys:
            to_return.append(self.attr_map_w[key])
        # need to transform to N arrays, as KERAS requires all labels for one output/attribute
        # in single array, so for 5 attributes and bulk 1024, it will be 5 arrays of length
        # 10240
        return [np.array(tmp_arr) for tmp_arr in zip(*to_return)]

    def get_images_online_w(self, img_names):
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
                path = IMAGES_FOLDER_WIKI + img_name
                # print(path)
                img = get_image(path, self.img_shape)
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                images.append(x)
            except Exception as e:
                print(path, str(e))
                errs.append(img_name)

        return np.vstack(images), errs

    def generate_training(self):
        return self.generate_data_m(self.train_ids_w, self.train_ids)

    def generate_validation(self):
        return self.generate_data_m(self.validation_ids_w, self.validation_ids)

    def generate_testing(self):
        return self.generate_data_m(self.test_ids_w, self.test_ids)
