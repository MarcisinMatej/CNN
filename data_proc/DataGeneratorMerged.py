from data_proc.DataGeneratorCelebA import DataGeneratorCelebA
from data_proc.DataGeneratorCelebASparse import create_map
from data_proc.DataGeneratorWiki import load_config_wiki
from data_proc.ConfigLoaderCelebA import load_attr_vals_txts, load_atributes_txts
import numpy as np
from keras.preprocessing import image

from data_proc.ImageHandler import get_image
from data_proc.ImagePreProcess import load_crop_boxes

# IMAGES_FOLDER_WIKI = "data_proc/data/"
IMAGES_FOLDER_CELEB = "data_proc/CelebA/img_align_celeba/"
CONF_FILE_WIKI = "wiki_cat_merged.txt"
CONF_FILE_IMDB = "imdb.txt"
IMAGES_FOLDER_IMDB = "data_proc/data/imdb/"

def create_map_m(attr_vals):
    """
    Helper method for loading attributes values from file.
    :param attr_vals: Raw data from file. List of string lines.
    :return: dictionary {name_of_image:list_of_ints}
    """
    _map = {}
    for attr_val in attr_vals:
        key = attr_val.split()[0].split("/")[-1]
        values = [i - 1 for i in list(map(int, attr_val.split()[1:]))]
        # add -1 for age
        values.append(-1)
        _map[key] = values
    return _map

def load_config_merged(conf_file):
    train = set()
    val = set()
    test = set()
    attr_map = {}
    with open("data_proc/config_files/" + conf_file, encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            arr = line.split("\t")
            key = arr[0]
            # if invalid_img(IMAGES_FOLDER_IMDB + key):
            #     continue
            gender_i = 0
            if arr[7] == 'F':
                gender_i = 1

            if int(arr[6]) < 22:
                age_cat = 0
            elif int(arr[6]) < 30:
                age_cat = 1
            elif int(arr[6]) < 40:
                age_cat = 2
            elif int(arr[6]) < 50:
                age_cat = 3
            elif int(arr[6]) < 60:
                age_cat = 4
            else:
                age_cat = 5

            attr_map[key] = [-1, -1, gender_i, -1, -1, age_cat]

            if arr[-1] == "1\n":
                train.add(key)
            if arr[-1] == "2\n":
                val.add(key)
            if arr[-1] == "3\n":
                test.add(key)

    print("---Training set has len: ", str(len(train)))
    print("---Testing set has len: ", str(len(test)))
    print("---Validation set has len: ", str(len(val)))
    return list(train), list(val), list(test), attr_map

class DataGeneratorMerged(DataGeneratorCelebA):
    def __init__(self, img_shape=(100, 100), chunk_size=1024):
        """

        :param img_shape: resolution of final image
        :param chunk_size: size of super batch
        :param rot_int: interval for image rotation
        """
        'Initialization'
        self.img_shape = img_shape
        self.chunk_size = chunk_size
        self.attr_map_celeb = create_map_m(load_atributes_txts())
        self.coord_dict = load_crop_boxes()
        # split data to training,testing,validation
        self.train_ids_imdb, self.validation_ids_imdb, self.test_ids_imdb, self.attr_map_imdb = load_config_merged(CONF_FILE_IMDB)

        self.train_ids = []
        self.test_ids = []
        self.validation_ids = []
        self.find_split_ids()
        self.img_source = IMAGES_FOLDER_CELEB

    def generate_data_m(self, pict_ids_imdb, pict_ids_c):
        """
        Generates data with hiding attributes according to MASKs
        :param pict_ids: ids of pictures
        :return:
        """
        # Generate Wiki data
        print("Generating W")
        indx = 0
        to = indx + self.chunk_size
        while indx <= len(pict_ids_imdb):
            images, errs = self.get_images_online_imdb(pict_ids_imdb[indx: to])
            if len(errs) > 0:
                # get only labels for images which were correctly loade
                img_labels = self.get_raw_labs_imdb(
                    [name for name in pict_ids_imdb[indx: to] if name not in errs])
            else:
                img_labels = self.get_raw_labs_imdb(pict_ids_imdb[indx: to])
            # get next boundaries
            to += self.chunk_size
            indx += self.chunk_size
            if to != len(pict_ids_imdb) and (indx + self.chunk_size) > len(pict_ids_imdb):
                # chunk increase overflow, we need to get the last chunk of data, which is smaller then defined
                to = len(pict_ids_imdb)

            yield images, img_labels
        print("Generating A")
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
        # need to transform to N arrays, as KERAS requires all labels for one output/attribute
        # in single array, so for 5 attributes and bulk 1024, it will be 5 arrays of length
        # 10240
        return [np.array(tmp_arr) for tmp_arr in zip(*to_return)]

    def get_raw_labs_imdb(self, keys):
        """
       Generate raw labels from attribute file for list of keys.
       :param keys: list of labels in string format
       :return: labels for specific batch of data in one-hot encoded format
       """
        to_return = []
        for key in keys:
            to_return.append(self.attr_map_imdb[key])
        # need to transform to N arrays, as KERAS requires all labels for one output/attribute
        # in single array, so for 5 attributes and bulk 1024, it will be 5 arrays of length
        # 10240
        return [np.array(tmp_arr) for tmp_arr in zip(*to_return)]

    def get_images_online_imdb(self, img_names):
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
                path = IMAGES_FOLDER_IMDB + img_name
                # print(path)
                img = get_image(path, self.img_shape)
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                images.append(x)
            except Exception as e:
                # print(path, str(e))
                errs.append(img_name)

        print("Caught ", str(len(errs)), " errors, which is ", str(len(errs)/len(img_names)*100), "%")
        return np.vstack(images), errs

    def generate_training(self):
        return self.generate_data_m(self.train_ids_imdb, self.train_ids)

    def generate_validation(self):
        return self.generate_data_m(self.validation_ids_imdb, self.validation_ids)

    def generate_testing(self):
        return self.generate_data_m(self.test_ids_imdb, self.test_ids)
