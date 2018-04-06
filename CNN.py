from __future__ import print_function
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.layers import Dense
from keras.layers import Input, Flatten
from keras.models import Model, model_from_json
from keras.utils import plot_model

from data_proc.DataLoader import get_attributes_desc, get_category_names
# from main_plots import _plot_model

history_path = "histories/"
config_path_var = "data_proc/config_files/training_vars.npy"


def Conv2DBatchNormRelu(n_filter, w_filter, h_filter, inputs, ind):
    return Activation(activation='relu',name="ReLU_"+ind)(BatchNormalization(name="BatchNorm_"+ind)
                    (Conv2D(n_filter, (w_filter, h_filter), padding='valid',name="Conv2D_"+ind)(inputs)))

def define_network_with_BN(in_shape=(32, 32, 3), single_output_ind = None):
    """
    Creates model for CNN with Keras functional api.
    Model graph is fixed and before each activation function is
    normalization layer. The output layers are defined by
    'attributes_values.txt' file. The number of lines is number of outputs
    and number of categories is determined by number of listed categories for each
    attribute.
    :param in_shape: tuple (x,y,z) of input shape for model
    :return: keras model
    """

    channel = "channels_last"
    #define shape of input
    inputs = Input(shape=in_shape,name="input")

    # define layers relationships
    conv1 = Conv2DBatchNormRelu(32, 3, 3, inputs=inputs,ind='1')
    conv2 = Conv2DBatchNormRelu(32,3, 3, inputs=conv1,ind='2')
    maxp1 = MaxPooling2D(pool_size=(2, 2), strides=2, data_format=channel)(conv2)
    conv3 = Conv2DBatchNormRelu(64, 3, 3, inputs=maxp1,ind='3')
    maxp2 = MaxPooling2D(pool_size=(2, 2), strides=2, data_format=channel)(conv3)
    conv4 = Conv2DBatchNormRelu(64, 3, 3, inputs=maxp2,ind='4')
    maxp3 = MaxPooling2D(pool_size=(2, 2), strides=2, data_format=channel)(conv4)
    conv5 = Conv2DBatchNormRelu(128, 3, 3, inputs=maxp3,ind='5')
    conv6 = Conv2DBatchNormRelu(128, 4, 4, inputs=conv5,ind='6')
    conv7 = Conv2DBatchNormRelu(2048, 5, 5, inputs=conv6,ind='7')
    conv8 = Conv2DBatchNormRelu(2048, 1, 1, inputs=conv7,ind='8')
    flatten = Flatten()(conv8)
    output_layers = []

    atrs_desc = get_attributes_desc()
    cat_names = get_category_names()
    for cnt, _name in zip(atrs_desc, cat_names):
        output_layers.append(Dense(cnt, activation='softmax', name=_name)(flatten))
    #TODO here we must also change to single output
    if single_output_ind is None:
        model = Model(inputs=inputs, outputs=output_layers)
    else:
        model = Model(inputs=inputs, outputs=output_layers[single_output_ind])
    # summarize layers
    print(model.summary())
    #TODO plot model
    # _plot_model(model)
    plot_model(model, to_file='model.png')
    return model


def define_network(in_shape=(32, 32, 3), single_output_ind = None):
    """
    Creates model for CNN with Keras functional api.
    Model graph is fixed. The output layers are defined by
    'attributes_values.txt' file. The number of lines is number of outputs
    and number of categories is determined by number of listed categories for each
    attribute.
    :param in_shape: tuple (x,y,z) of input shape for model
    :return: keras model
    """

    channel = "channels_last"
    #define shape of input
    inputs = Input(shape=in_shape,name="input")

    # define layers relationships
    conv1 = Conv2D(32, (3, 3), padding='valid',activation='relu')(inputs)
    conv2 = Conv2D(32, (3, 3), padding='valid',activation='relu')(conv1)
    maxp1 = MaxPooling2D(pool_size=(2, 2), strides=2,data_format=channel)(conv2)
    conv3 = Conv2D(64, (3, 3), padding='valid',activation='relu')(maxp1)
    maxp2 = MaxPooling2D(pool_size=(2, 2), strides=2,data_format=channel)(conv3)
    conv4 = Conv2D(64, (3, 3), padding='valid',activation='relu')(maxp2)
    maxp3 = MaxPooling2D(pool_size=(2, 2), strides=2,data_format=channel)(conv4)
    conv5 = Conv2D(128, (3, 3), padding='valid',activation='relu')(maxp3)
    conv6 = Conv2D(128, (4, 4), padding='valid',activation='relu')(conv5)
    conv7 = Conv2D(2048, (5, 5), padding='valid',activation='relu')(conv6)
    conv8 = Conv2D(2048, (1, 1), padding='valid',activation='relu')(conv7)
    flatten = Flatten()(conv8)
    output_layers = []
    # output layers
    # out1 = Dense(2, activation='softmax',name="out1")(flatten)
    # output_layers.append(out1)
    # out2 = Dense(2, activation='softmax',name="out2")(flatten)
    # output_layers.append(out2)
    # out3 = Dense(2, activation='softmax',name="out3")(flatten)
    # output_layers.append(out3)
    # out4 = Dense(2, activation='softmax',name="out4")(flatten)
    # output_layers.append(out4)
    # out5 = Dense(5, activation='softmax',name="out5")(flatten)
    # output_layers.append(out5)

    atrs_desc = get_attributes_desc()
    cat_names = get_category_names()
    for cnt, _name in zip(atrs_desc, cat_names):
        output_layers.append(Dense(cnt, activation='softmax', name=_name)(flatten))
    #TODO here we must also change to single output
    if single_output_ind is None:
        model = Model(inputs=inputs, outputs=output_layers)
    else:
        model = Model(inputs=inputs, outputs=output_layers[single_output_ind])
    # summarize layers
    print(model.summary())
    #TODO plot model
    # _plot_model(model)
    plot_model(model, to_file='model.png')
    return model

def save_model(model,path,ep_ind,best_loss,best_ep_ind):
    """
    Saves KERAS model to designated location. Model is saved
    as json_file and weights are separately in model.h5 file
    :param model: model to save
    :param path: location to save model
    :return:
    """
    with open(path + "model.json", "w") as json_file:
        json_file.write(model.to_json())
    # serialize weights to HDF5
    model.save_weights(path + "model.h5")
    save_vars(ep_ind,best_loss,best_ep_ind)
    print("Saved model to disk")



def load_model(path):
    """
    Loads KERAS model from designated location. Model is loaded
    from json_file and weights are separately loaded from model.h5 file.
    Location of config dictionary is config files directory.
    :param path: path to folder with 'model.json' file and 'model.h5' files.
    :return: loaded model with weights,dictionary of saved config variables (eg.epoch index, best validation loss)
    For more see save_vars()
    """
    # load json and create model
    json_file = open(path+'model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path+"model.h5")
    print("Loaded model from disk")
    return loaded_model,load_config_dict(config_path_var)


def serialize_history(dict,ep_ind):
    """
    Save history(dictionary) to disk.
    :param dict:
    :param ep_ind:
    :return:
    """
    # Save
    save_dictionary(history_path + str(ep_ind) + "_hist", dict)


def load_dictionary(path_loc):
    """
    Loads dictionary from specified path.
    :param path_loc:
    :return:
    """
    # Save
    return np.load(path_loc).item()


def save_dictionary(path_loc, dict):
    """
    Saves dictionary to specified path into npy file.
    :param dict:
    :param path_loc:
    :return:
    """
    # Save
    np.save(path_loc + ".npy", dict)


def save_vars(ep_ind, best_loss,best_ep_ind):
    """
    Saves parameters of model run, like current epoch index...
    :param best_ep_ind: index of epoch with the best val. loss
    :param ep_ind: current epoch index
    :param best_loss: currently the best validation loss
    :return:
    """
    dict = {}
    dict['epoch'] = ep_ind
    dict['loss'] = best_loss
    dict['ep_ind'] = best_ep_ind
    np.save(config_path_var, dict)


def load_config_dict(config_path_var):
    """
    In case dictionary was
    not serialized before the basic initial values are returned.
    :param config_path_var:
    :return:
    """
    try:
        return load_dictionary(config_path_var)
    except Exception as e:
        var_dict = {}
        var_dict['epoch'] = 0
        var_dict['loss'] = float("inf")
        var_dict['ep_ind'] = 0
        return var_dict
