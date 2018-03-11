from keras.utils import np_utils

from CNN import load_dictionary
from keras.preprocessing import image

from data_proc.DataGenerator import data_folder
from data_proc.DataLoader import  load_folder_txts, load_attr_vals_txts
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import random

from data_proc.ImageParser import get_crop_resize
from data_proc.ImagePreProcess import load_crop_boxes

LABEL_DICT_PATH = "data_proc/encoded_labels.npy"
IMAGES_FOLDER = "data_proc/CelebA/img_align_celeba/"
VIRT_GEN_STOP = 10


def scale_coords(coords, scale):
    sc_coords = []
    sc_coords.append(int(coords[0] + (-1)*scale*coords[0]/100))
    sc_coords.append(int(coords[1] + (-1) * scale * coords[1]/100))
    sc_coords.append(int(coords[2] + scale * coords[2]/100))
    sc_coords.append(int(coords[3] + scale * coords[3]/100))
    return sc_coords


class MyVirtualGenerator(object):
    """Generates data for Keras"""
    def __init__(self, img_shape=(100, 100), chunk_size=1024, rot_int=(-20, 20), sc_int=(-5, 5), virt_dupl=5):
        """

        :param img_shape: resolution of final image
        :param chunk_size: size of super batch
        :param rot_int: interval for image rotation
        """
        'Initialization'
        self.img_shape = img_shape
        self.chunk_size = chunk_size
        self.attr_vals = load_attr_vals_txts()
        self.attr_map = load_dictionary(LABEL_DICT_PATH)
        self.coord_dict = load_crop_boxes()
        self.rotation_interval = rot_int
        self.scale_interval = sc_int
        self.virt_dupl = virt_dupl
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
        self.rng = random.SystemRandom()

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
        for key in keys:
            to_return.append(self.attr_map[key])
        # need to transform to N arrays, as KERAS requires all labels for one output/attribute
        # in single array, so for 5 attributes and bulk 1024, it will be 5 arrays of length
        # 1024
        return [np.array(tmp_arr) for tmp_arr in zip(*to_return)]

    def generate_data(self, names):
        for v_ep in range(self.virt_dupl):
            print("-->Virtual run [", str(v_ep), "]")
            i = 0
            while (i + self.chunk_size) < len(names):
                # img_labels = self.get_encoded_labels(names[i:i + self.chunk_size])
                images, errs = self.get_transformed_images(names[i:i+self.chunk_size])
                if len(errs) > 0:
                    img_labels = self.get_encoded_labels([name for name in names[i:i + self.chunk_size] if name not in errs])
                else:
                    img_labels = self.get_encoded_labels(names[i:i + self.chunk_size])
                i += self.chunk_size
                yield images, img_labels

            #yield the rest of images
            if i < len(names):
                images, errs = self.get_transformed_images(names[i:len(names)])
                if len(errs) > 0:
                    print("ERROR reading images, removing name from labels")
                    img_labels = self.get_encoded_labels(
                        [name for name in names[i:i + self.chunk_size] if name not in errs])
                else:
                    img_labels = self.get_encoded_labels(names[i:i + self.chunk_size])
                yield images, img_labels

    def generate_data_eval(self, names, folder):
        i = 0
        while (i + self.chunk_size) < len(names):
            # img_labels = self.get_encoded_labels(names[i:i + self.chunk_size])
            images, errs = self.load_images(names[i:i+self.chunk_size], folder)
            # remove error labels
            if len(errs) > 0:
                print("ERROR reading images, removing name from labels")
                img_labels = self.get_encoded_labels([name for name in names[i:i+self.chunk_size] if name not in errs])
            else:
                img_labels = self.get_encoded_labels(names[i:i + self.chunk_size])
            i += self.chunk_size
            yield images, img_labels

        #yield the rest of images
        if i < len(names):
            images, errs = self.load_images(names[i:len(names)], folder)
            if len(errs) > 0:
                print("ERROR reading images, removing name from labels")
                img_labels = self.get_encoded_labels([name for name in names[i:i+self.chunk_size] if name not in errs])
            else:
                img_labels = self.get_encoded_labels(names[i:i + self.chunk_size])
            yield images, img_labels

    def generate_training(self):
        return self.generate_data(self.train_ids)

    def generate_validation(self):
        return self.generate_data_eval(self.validation_ids,'validation/')

    def generate_testing(self):
        return self.generate_data_eval(self.test_ids,'test/')

    def load_images(self,img_names,folder):
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
        errs = []
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
                errs.append(img_name)

        return np.vstack(images),errs

    def get_transformed_images(self, img_names):
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
                path = IMAGES_FOLDER + img_name
                # print(path)
                img = get_crop_resize(path,
                                      scale_coords(self.coord_dict[img_name], self.rng.randint(*self.scale_interval)),
                                      self.rng.randint(*self.rotation_interval),
                                      self.img_shape)
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                images.append(x)
            except Exception as e:
                print(str(e))
                errs.append(img_name)

        return np.vstack(images), errs

    def generate_data_labeled(self):
        """
        Returns chunks of pictures with pictures names for virtual generator.
        :return:
        """
        i = 0
        while (i + self.chunk_size) < len(self.train_ids):
            images = self.get_transformed_images(self.train_ids[i:i+self.chunk_size])
            i += self.chunk_size
            yield images, self.train_ids[i:i+self.chunk_size]

        # yield the rest of images
        if i < len(self.train_ids):
            img_labels = self.get_encoded_labels(self.train_ids[i:len(self.train_ids)])
            images = self.get_transformed_images(self.train_ids[i:len(self.train_ids)])
            yield images, img_labels

    def virtual_train_generator(self):
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # only needed in case of feturewise_center/whitening...
        # datagen.fit(X_sample)  # let's say X_sample is a small-ish but statistically representative sample of your data

        # TODO possible add limit
        train_gen = self.generate_data_labeled()
        # training
        for X_train, Y_train in train_gen:  # these are chunks of ~bulk pictures
            gen_cnt = 0
            for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size=self.chunk_size):  # these are chunks of virtualized immages
                gen_cnt += 1
                print("GENERATOR [" + str(gen_cnt) + "]")
                if gen_cnt >= VIRT_GEN_STOP:
                    # we need to break the loop by hand because
                    # the generator loops indefinitely
                    print("BREAKING GENERATOR.")
                    break
                yield X_batch, self.get_encoded_labels(Y_batch)