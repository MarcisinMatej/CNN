import numpy as np
from PIL import Image
from scipy import ndimage

PATH = 'data_proc/config_files/'


def load_folder_txts():
    with open(PATH+'folder.txt') as file_folder:
        folder = file_folder.readlines()
    return folder


def load_attr_vals_txts():
    with open(PATH+'attribute_values.txt') as file_attr_vals:
        return file_attr_vals.readlines()


def load_label_txts():
    with open(PATH+'attribute_values.txt') as file_attr_vals:
        attr_vals = file_attr_vals.readlines()
    with open(PATH+'attributes.txt') as file_attrs:
        attrs = file_attrs.readlines()
    attr_map = {}
    for attr in attrs:
        attr_map[attr.split()[0].split("/")[-1]] = attr.split()[1:]
    return attr_vals, attr_map


def load_atributes_txts():
    with open(PATH+'attributes.txt') as file_attrs:
        return file_attrs.readlines()


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


def get_attributes_desc():
    # count how many classes are in each label (length of one-hot true value vector)
    atrs_desc = []
    with open(PATH + 'attribute_values.txt') as file_attr_vals:
        for line in file_attr_vals.readlines():
            cnt = len(line.split(":")[-1].split(","))
            atrs_desc.append(cnt)
    return atrs_desc

def get_category_names():
    """
    :return: list of names of categories
    """
    atrs_names = []
    with open(PATH + 'attribute_values.txt') as file_attr_vals:
        for line in file_attr_vals.readlines():
            atrs_names.append(line.split(":")[0].split("-")[-1])
    return atrs_names

def get_cat_attributes_names():
    """
    :return: 2D array, in row we have attributes values
    """
    cat_atrs_names = []
    with open(PATH + 'attribute_values.txt') as file_attr_vals:
        for line in file_attr_vals.readlines():
            cat_atrs_names.append(line.split(":")[-1].split(","))
    return cat_atrs_names


def flatten_image(image_path):
    '''
    Flattens image to 3D flat vector
    :param image_path: path to image location
    :return: single image flatten in format channel first ( eg. picture 32x32 in rgb will be 3x1024)
    '''
    ndimage.imread(image_path).transpose((2, 0, 1))
