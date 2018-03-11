from keras.utils import np_utils

from CNN import load_dictionary
from data_proc.DataGenerator import DataGenerator
from data_proc.DataLoader import  load_folder_txts, load_attr_vals_txts

LABEL_DICT_PATH = "encoded_labels.npy"

class DataGenerator2(DataGenerator):
    """Generates data for Keras"""
    def __init__(self, img_shape=(32,32), chunk_size=32):
        'Initialization'
        self.img_shape = img_shape
        self.chunk_size = chunk_size
        self.attr_vals = load_attr_vals_txts()
        self.attr_map = load_dictionary(LABEL_DICT_PATH)
        # count how many different attributes we will predict
        self.attr_cnt = len(self.attr_vals)
        self.attr_class_cnt = []
        # count how many classes are in each label (length of one-hot true value vector)
        for attr_val in self.attr_vals:
            cnt = len(attr_val.split(":")[1].split(","))
            self.attr_class_cnt.append(cnt)
        # split data to training,testing,validation
        self.train_ids = []
        self.test_ids = []
        self.validation_ids = []

        self.find_split_ids()

    def find_split_ids(self):
        """
        Finds ids of training,testing and validation data from config file folders.txt
        :return:
        """
        folder = load_folder_txts()
        for line in folder:
            i = line.split()[-1]
            if i == "1":
                self.train_ids.append(line.split()[0].split("/")[-1])
            elif i == "2":
                self.test_ids.append(line.split()[0].split("/")[-1])
            elif i == "3":
                self.validation_ids.append(line.split()[0].split("/")[-1])
        print("Done")

