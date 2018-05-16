from data_proc.DataGeneratorOnLine import DataGeneratorOnLine
from data_proc.DataLoaderCelebA import load_attr_vals_txts
from keras.preprocessing import image
import numpy as np

from data_proc.ImageParser import invalid_img, get_image

# IMAGES_FOLDER_IMDB = "/datagrid/personal/marcisin/"
IMAGES_FOLDER_IMDB = "data_proc/data/imdb/"
CONF_FILE = "imdb.txt"

def load_config_imdb(conf_file):
    train = set()
    val = set()
    test = set()
    attr_map = {}
    coord_dict = {}
    with open("data_proc/config_files/"+conf_file, encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            arr = line.split("\t")
            key = arr[0]
            # if invalid_img(IMAGES_FOLDER_IMDB + key):
            #     continue
            coord_dict[key] = DataGeneratorOnLine.expand_coords(list(map(int, arr[2:6])))
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

            attr_map[key] = [gender_i, age_cat]

            if arr[-1] == "1\n":
                train.add(key)
            if arr[-1] == "2\n":
                val.add(key)
            if arr[-1] == "3\n":
                test.add(key)

    print("---Training set has len: ", str(len(train)))
    print("---Testing set has len: ", str(len(test)))
    print("---Validation set has len: ", str(len(val)))
    return list(train), list(val), list(test), attr_map, coord_dict


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
        # count how many different attributes we will predict
        self.attr_cnt = len(self.attr_vals)
        # split data to training,testing,validation
        self.train_ids, self.validation_ids, self.test_ids, self.attr_map, self.coord_dict = load_config_imdb(CONF_FILE)
        self.img_source = IMAGES_FOLDER_IMDB
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