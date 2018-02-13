import numpy as np
from keras.preprocessing import image
from keras.utils import np_utils

from data_proc.DataLoader import load_label_txts, load_folder_txts

data_folder = 'data_proc/data/'

class DataGenerator(object):
    """Generates data for Keras"""
    def __init__(self, img_shape=(32,32), chunk_size=32):
        'Initialization'
        self.img_shape = img_shape
        self.chunk_size = chunk_size
        self.attr_vals, self.attr_map = load_label_txts()
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

        self.find_offsets()

    def find_offsets(self):
        #TODO nacist si tri pole indexov trenovaci/testovaci/validacni
        """
        Finds offset of training,testing and validation data from config file folders.txt
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

        # # marker splits data to 3 parts, training-1/testing-2/validation-3
        # while folder[i].split()[-1] != "3":
        #     if self.test_offset == 0 and folder[i].split()[-1] == "2":
        #         self.test_offset = i
        #     i+=1
        # self.validation_offset = i

    def labels_generator(self, list):
        """
        Generate labels from attribute file
        :param curr_offset: starting position in attribute list
        :return: labels for specific batch of data
        """
        img_labels =[[] for i in range(self.attr_cnt)]
        #create dictionary of image name and attributes labels
        for entry_id, ind in zip(list,range(len(list))):
            line_arr = self.attr_map[entry_id]
            #create vectors of ints for each entry of attribute values
            for att_ind in range(self.attr_cnt):
                # -1 becasue in config file we count from 1
                # image i attribute j
                img_labels[att_ind].append(int(line_arr[att_ind]) - 1)

        # convert to one hot vector
        to_return = []
        for attr_ind in range(self.attr_cnt):
            to_return.append(np_utils.to_categorical(img_labels[attr_ind], self.attr_class_cnt[attr_ind]))

        return to_return
        # print(attr_class_cnt)

    def generate_data(self, names, folder):
        i = 0
        # TODO solve overlap, now it is irrelevant
        while (i + self.chunk_size) < len(names):
            # try catch for possible problems with image formats
            try:
                # b_i is index in batch
                img_labels = self.labels_generator(names[i:i+self.chunk_size])
                images = self.get_transformed_images(names[i:i+self.chunk_size], folder)
                i += self.chunk_size
                yield images, img_labels
            except Exception as e:
                print(str(e))
                i += self.chunk_size

    def generate_training(self):
        return self.generate_data(self.train_ids,'train/')

    def generate_testing(self):
        return self.generate_data(self.test_ids,'test/')

    def get_transformed_images(self,img_names,folder):
        images = []
        for img_name in img_names:
            path = data_folder + folder + img_name
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
