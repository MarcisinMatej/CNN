import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

from CNN import load_dictionary
from data_proc.DataLoader import load_label_txts, load_folder_txts, load_attr_vals_txts

data_folder = 'data_proc/data/'

LABEL_DICT_PATH = "data_proc/encoded_labels.npy"

class DataGenerator(object):
    """Generates data for Keras"""
    def __init__(self, img_shape=(100, 100), chunk_size=1024, on_line_labels=True):
        'Initialization'
        self.img_shape = img_shape
        self.chunk_size = chunk_size
        self.on_line_lbls = on_line_labels
        if on_line_labels:
            self.attr_vals, self.attr_map = load_label_txts()
        else:
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

    def generate_all_encoded_labels(self):
        all_imgs = self.attr_map.keys()
        return all_imgs, self.get_encoded_labels(all_imgs)

    def get_encoded_labels(self, keys):
        """
        Generate labels from attribute file for list of keys,
        the labels are returned in the same order as corresponding
        keys in parameter list.
        :param keys: list of labels in string format
        :return: labels for specific batch of data in one-hot encoded format
        """
        to_return = []
        if self.on_line_lbls:
            img_labels =[[] for _ in range(self.attr_cnt)]
            # create dictionary of image name and attributes labels
            for entry_id, ind in zip(keys, range(len(keys))):
                line_arr = self.attr_map[entry_id]
                # create vectors of ints for each entry of attribute values
                for att_ind in range(self.attr_cnt):
                    # -1 because in config file we count from 1
                    # image i attribute j
                    img_labels[att_ind].append(int(line_arr[att_ind]) - 1)

            # convert to one hot vector
            to_return = []
            for attr_ind in range(self.attr_cnt):
                to_return.append(np_utils.to_categorical(img_labels[attr_ind], self.attr_class_cnt[attr_ind]))
            # to_return = [list(i) for i in zip(*tmp)]
        else:
            tmp = []
            for key in keys:
                tmp.append(self.attr_map[key])
            to_return = [np.array(tmp_arr) for tmp_arr in zip(*tmp)]
        return to_return

    def generate_data(self, names, folder):
        i = 0
        while (i + self.chunk_size) < len(names):
            img_labels = self.get_encoded_labels(names[i:i + self.chunk_size])
            images = self.get_transformed_images(names[i:i+self.chunk_size], folder)
            i += self.chunk_size
            yield images, img_labels

        #yield the rest of images
        if i < len(names):
            img_labels = self.get_encoded_labels(names[i:len(names)])
            images = self.get_transformed_images(names[i:len(names)], folder)
            yield images, img_labels

    def generate_training(self):
        return self.generate_data(self.train_ids,'train/')

    def generate_validation(self):
        return self.generate_data(self.validation_ids,'validation/')

    def generate_testing(self):
        return self.generate_data(self.test_ids,'test/')

    def get_transformed_images(self,img_names,folder):
        """
        Reads list of images from specidied folder.
        The images are resized to self.img_shape specified
        in the generator contructor.
        In case of error, image is not added to return list
        and error is just printed.
        :param img_names: List of image names
        :param folder: Source folder
        :return: list of vstacked images, channel_last format
        """
        images = []
        for img_name in img_names:
            try:
                path = data_folder + folder + img_name
                # print(path)
                img = image.load_img(path, target_size=self.img_shape)
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                images.append(x)
            except Exception as e:
                print(str(e))

        return np.vstack(images)

    def generate_data_labeled(self):
        """
        Returns chunks of pictures with pictures names for virtual generator.
        :return:
        """
        folder = "train/"
        i = 0
        while (i + self.chunk_size) < len(self.train_ids):
            images = self.get_transformed_images(self.train_ids[i:i+self.chunk_size], folder)
            i += self.chunk_size
            yield images, self.train_ids[i:i+self.chunk_size]

        # yield the rest of images
        if i < len(self.train_ids):
            img_labels = self.get_encoded_labels(self.train_ids[i:len(self.train_ids)])
            images = self.get_transformed_images(self.train_ids[i:len(self.train_ids)], folder)
            yield images, img_labels

# for debug purposes
if __name__ == "__main__":
    # run_data_crop()
    tmp = DataGenerator()
    gen = tmp.generate_training()
    cnt = 0
    for i in gen:
        if cnt < 1 :
            print(i[0].shape)
            print('[1]',i[1][0])
            print("===============================================")
            print('[2]', i[1][1])
            print("===============================================")
            print('[3]', i[1][2])
            print("===============================================")
            print('[4]', i[1][3])
            print("===============================================")
            print('[5]', i[1][4])
        cnt+=1

    # test_datagen = ImageDataGenerator(rescale=1. / 255)
    #
    # validation_generator = test_datagen.flow_from_directory('data',target_size=(256, 256), batch_size=32,class_mode = 'categorical')
    # validation_generator.next()
    # print(len(validation_generator.filenames))
