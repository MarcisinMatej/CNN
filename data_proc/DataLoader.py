import numpy as np
from PIL import Image
from scipy import ndimage

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
    :return: single image flatten in format channel first ( eg. picture 32x32 in rgb will be 3x1024)
    '''
    ndimage.imread(image_path).transpose((2, 0, 1))
