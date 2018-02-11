import numpy as np
from keras.preprocessing import image
from keras.utils import np_utils

from data_proc.DataLoader import load_label_txts, load_folder_txts


class DataGenerator(object):
    """Generates data for Keras"""
    def __init__(self, img_shape=(32,32), chunk_size=32):
        'Initialization'
        self.img_shape = img_shape
        self.chunk_size = chunk_size
        self.attr_vals, self.attrs = load_label_txts()
        # count how many different attributes we will predict
        self.attr_cnt = len(self.attr_vals)
        self.attr_class_cnt = []
        # count how many classes are in each label (length of one-hot true value vector)
        for attr_val in self.attr_vals:
            cnt = len(attr_val.split(":")[1].split(","))
            self.attr_class_cnt.append(cnt)
        # split data to training,testing,validation
        self.train_offset = 0
        self.test_offset = 0
        self.validation_offset = 0
        self.find_offsets()

    def find_offsets(self):
        #TODO nacist si tri pole indexov trenovaci/testovaci/validacni
        """
        Finds offset of training,testing and validation data from config file folders.txt
        :return:
        """
        i = 0
        folder = load_folder_txts()

        # marker splits data to 3 parts, training-1/testing-2/validation-3
        while folder[i].split()[-1] != "3":
            if self.test_offset == 0 and folder[i].split()[-1] == "2":
                self.test_offset = i
            i+=1
        self.validation_offset = i

    def labels_generator(self, curr_offset):
        """
        Generate labels from attribute file
        :param curr_offset: starting position in attribute list
        :return: labels for specific batch of data
        """
        img_labels =[[None for i in range(self.chunk_size)] for j in range(self.attr_cnt)]
        #create dictionary of image name and attributes labels
        for entry_ind in range(curr_offset, curr_offset + self.chunk_size):
            attr_line = self.attrs[entry_ind]
            line_arr = attr_line.split()
            img_name = line_arr[0].split("/")[-1]
            #create vectors of ints for each entry of attribute values
            for att_ind in range(self.attr_cnt):
                attr_value = int(line_arr[att_ind+1]) - 1
                # immage i attribute j
                img_labels[att_ind][entry_ind%self.chunk_size] = attr_value

        # convert to one hot vector
        to_return = []
        for attr_ind in range(self.attr_cnt):
            to_return.append(np_utils.to_categorical(img_labels[attr_ind], self.attr_class_cnt[attr_ind]))

        return to_return
        # print(attr_class_cnt)

    def generate_data(self, start_offset, end_offset):
        i = start_offset
        # TODO solve overlap, now it is irrelevant
        while (i + self.chunk_size) < end_offset:
            # try catch for possible problems with image formats
            try:
                # b_i is index in batch
                img_labels = self.labels_generator(i)
                images = self.get_transformed_images(i)
                i += self.chunk_size
                yield images, img_labels
            except Exception as e:
                print(str(e))
                i += self.chunk_size

    def generate_training(self):
        return self.generate_data(self.train_offset, self.test_offset)

    def generate_testing(self):
        return self.generate_data(self.test_offset, self.validation_offset)

    def get_transformed_images(self, curr_offset):
        images = []
        for i in range(curr_offset, curr_offset + self.chunk_size):
            path = 'data_proc/data/train/' + self.attrs[i].split()[0].split("/")[-1]
            # print(path)
            img = image.load_img(path, target_size=self.img_shape)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            images.append(x)

        return np.vstack(images)


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
