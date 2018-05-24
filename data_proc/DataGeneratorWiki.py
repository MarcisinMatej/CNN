from data_proc.ConfigLoaderWiki import load_config_wiki
from data_proc.DataGeneratorCelebA import DataGeneratorCelebA
from data_proc.ConfigLoaderCelebA import load_attr_vals_txts
from data_proc.ImageHandler import get_image
from keras.preprocessing import image
import numpy as np

IMAGES_FOLDER = "data_proc/data/wiki_crop/"


class DataGeneratorWiki(DataGeneratorCelebA):
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
        self.train_ids, self.validation_ids, self.test_ids, self.attr_map = load_config_wiki()
        print("-- Generator Wiki initialized.")

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
                path = IMAGES_FOLDER + img_name
                # print(path)
                img = get_image(path, self.img_shape)
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                images.append(x)
            except Exception as e:
                # print(path,str(e))
                errs.append(img_name)

        return np.vstack(images), errs

    def generate_data(self, pict_ids):
        """
                Generates data with hiding attributes according to MASKs
                :param pict_ids: ids of pictures
                :return:
                """
        indx = 0
        to = indx + self.chunk_size
        while indx <= len(pict_ids):
            images, errs = self.get_images_online(pict_ids[indx: to])
            if len(errs) > 0:
                # get only labels for images which were correctly loade
                img_labels = self.get_encoded_labels(
                    [name for name in pict_ids[indx: to] if name not in errs])
            else:
                img_labels = self.get_encoded_labels(pict_ids[indx: to])
            # get next boundaries
            to += self.chunk_size
            indx += self.chunk_size
            if to != len(pict_ids) and (indx + self.chunk_size) > len(pict_ids):
                # chunk increase overflow, we need to get the last chunk of data, which is smaller then defined
                to = len(pict_ids)

            yield images, img_labels

    def get_encoded_labels(self, keys):
        """
        Generate labels from attribute file for list of keys,
        the labels are returned in the same order as corresponding
        keys in parameter list.
        :param keys: list of labels in string format
        :return: labels for specific batch of data in raw form.
        """
        to_return = []
        for key in keys:
            to_return.append(self.attr_map[key])
        # need to transform to N arrays, as KERAS requires all labels for one output/attribute
        # in single array, so for 5 attributes and bulk 1024, it will be 5 arrays of length
        # 10240
        return [np.array(tmp_arr) for tmp_arr in zip(*to_return)]