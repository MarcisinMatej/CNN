from data_proc.image_parser import *
import numpy as np
from scipy import ndimage
from PIL import Image
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image


PATH = 'data_proc/config_files/'

def load_folder_txts():
    with open(PATH+'folder.txt') as file_folder:
        folder = file_folder.readlines()
    return folder


def load_label_txts():
    with open(PATH+'attribute_values.txt') as file_attr_vals:
        attr_vals = file_attr_vals.readlines()
    with open(PATH+'attributes.txt') as file_attrs:
        attrs = file_attrs.readlines()
    return attr_vals, attrs

def load_crop_txts():
    with open(PATH+'bboxes.txt') as file_bboxes:
        bboxes = file_bboxes.readlines()
    with open(PATH+'folder.txt') as file_folder:
        folder = file_folder.readlines()
    return bboxes, folder

def crop_store_images(bboxes, folders):
    images = []
    for bbox,folder in zip(bboxes,folders):
        arr = bbox.split()
        img = {
            'path': arr[0],
            'bbox': (float(arr[1]), float(arr[2]), float(arr[5]), float(arr[6])),
            'name': arr[0].split('/')[-1]
        }
        code = int(folder.split()[-1])
        #parse output folder
        if code==1:
            img.update({'folder':'data/train/'})
        elif code==2:
            img.update({'folder':'data/test/'})
        if code==3:
            img.update({'folder':'data/validation/'})
        images.append(img)
    for img in images:
        # TODO change here we crop and resize
        image_parser.crop_resize(img['path'], img['bbox'], img['folder'] + img['name'])

def run_data_crop():
    '''
    method processes images in such a manner, that it will cut out face and store it in separate folders (training,testing,validaiton)
    :return:
    '''
    bboxes, folder = load_crop_txts()
    crop_store_images(bboxes, folder)
    # CelebA/img_align_celeba/000001.jpg 45 76 148 76 148 179 45 179

def matrix_image(image):
    Standard_size = (32,32)
    "opens image and converts it to a m*n matrix"
    image = Image.open(image)
    print("changing size from %s to %s" % (str(image.size), str(Standard_size)))
    image = image.resize(Standard_size)
    image = list(image.getdata())
    image = map(list,image)
    image = np.array(image)
    return image

def flatten_image(image_path):
    '''
    Flattens image to 3D flat vector
    :param image_path: path to image location
    :return:
    '''
    ndimage.imread(image_path).transpose((2, 0, 1))


class DataGenerator(object):
    'Generates data for Keras'

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
        self.FindOffsets()


    def FindOffsets(self):
        i = 0
        folder = load_folder_txts()
        self.train_offset=0
        self.test_offset=0
        self.validation_offset = 0

        while folder[i].split()[-1] != "3":
            if self.test_offset == 0 and folder[i].split()[-1] == "2":
                self.test_offset = i
            i+=1
        self.validation_offset = i


    def Labels_generator(self,curr_offset):
        '''
        Generate labels from attribute file
        :param curr_offset: starting position in attribute list
        :return: labels for specific batch of data
        '''
        img_labels =[[None for i in range(self.chunk_size)] for j in range(self.attr_cnt)]
        #create dictionary of image name and attributes labels
        for i in range(curr_offset, curr_offset + self.chunk_size):
            attr_line = self.attrs[i]
            line_arr = attr_line.split()
            img_name = line_arr[0].split("/")[-1]
            #create one hot vectors for each attribute entry
            for j in range(self.attr_cnt):
                attr_value = int(line_arr[j+1]) - 1
                # immage i attribute j
                img_labels[j][i%self.chunk_size] = attr_value

        # convert to one hot vector
        to_return = []
        for i in range(self.attr_cnt):
            to_return.append(np_utils.to_categorical(img_labels[i], self.attr_class_cnt[i]))

        return to_return
        # print(attr_class_cnt)

    def TrainingGenerator(self):
        i = self.train_offset
        # TODO solve overlap, know it is irrelevant
        while i < self.test_offset:
            # b_i is index in batch
            img_labels = self.Labels_generator(i)
            images = self.get_transformed_images_training(i)
            i+=self.chunk_size
            yield images,img_labels

    def get_transformed_images_training(self, curr_offset):
        images = []
        for i in range(curr_offset, curr_offset + self.chunk_size):
            path = 'data_proc/data/train/' + self.attrs[i].split()[0].split("/")[-1]
            # print(path)
            img = image.load_img(path, target_size=self.img_shape)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            images.append(x)

        return np.vstack(images)

if __name__ == "__main__":
    # run_data_crop()
    tmp = DataGenerator()
    gen = tmp.TrainingGenerator()
    cnt=0
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
