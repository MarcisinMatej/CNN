from __future__ import print_function
import numpy as np
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense
from keras.layers import Input, Flatten
from keras.models import Model, model_from_json
from data_proc.DataLoader import get_attributes_desc

history_path = "histories/"
config_path_var = "data_proc/config_files/training_vars.npy"

def define_network(in_shape=(32, 32, 3)):
    """
    Creates model for CNN with Keras functional api.
    Model graph is fixed. The output layers are defined by
    'attributes_values.txt' file. The number of lines is number of outputs
    and number of categories is determined by number of listed categories for each
    attribute.
    :param in_shape: tuple (x,y,z) of input shape for model
    :return: keras model
    """

    #define shape of input
    input = Input(shape=in_shape,name="input")

    # define layers relationships
    conv1 = Conv2D(32, (3, 3), padding='same',activation='relu')(input)
    conv2 = Conv2D(32, (3, 3), padding='same',activation='relu')(conv1)
    maxp1 = MaxPooling2D(pool_size=(2, 2), strides=2,data_format="channels_first")(conv2)
    conv3 = Conv2D(64, (3, 3), padding='same',activation='relu')(maxp1)
    maxp2 = MaxPooling2D(pool_size=(2, 2), strides=2,data_format="channels_first")(conv3)
    conv4 = Conv2D(64, (3, 3), padding='same',activation='relu')(maxp2)
    maxp3 = MaxPooling2D(pool_size=(2, 2), strides=2,data_format="channels_first")(conv4)
    conv5 = Conv2D(128, (3, 3), padding='same',activation='relu')(maxp3)
    conv6 = Conv2D(128, (4, 4), padding='same',activation='relu')(conv5)
    conv7 = Conv2D(2048, (5, 5), padding='same',activation='relu')(conv6)
    conv8 = Conv2D(2048, (1, 1), padding='same',activation='relu')(conv7)
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
    for cnt, ind in zip(atrs_desc, range(len(atrs_desc))):
        _name = "out"+str(ind)
        output_layers.append(Dense(cnt, activation='softmax', name=_name)(flatten))
    #TODO here we must also change to single output
    model = Model(inputs=input, outputs=output_layers)
    # summarize layers
    print(model.summary())
    #TODO plot model
    # _plot_model(model)
    return model


def save_model(model,path,ep_ind,best_loss):
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
    save_vars(ep_ind,best_loss)
    print("Saved model to disk")

def load_model(path):
    """
    Loads KERAS model from designated location. Model is loaded
    from json_file and weights are separately loaded from model.h5 file.
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
    return loaded_model,load_dictionary(config_path_var)

def serialize_history(dict,ep_ind):
    """
    Save history(dictionary) so disk.
    :param dict:
    :param ep_ind:
    :return:
    """
    # Save
    np.save(history_path + str(ep_ind) + "_hist.npy", dict)


def load_dictionary(path_loc):
    """
    Loads dictionary from specified path.
    :param path_loc:
    :return:
    """
    # Save
    return np.load(path_loc).item()


def save_vars(ep_ind, best_loss):
    """
    Saves parameters of model run, like current epoch index...
    Location is config files directory.
    :param ep_ind: current epoch index
    :param best_loss: currently the best validation loss
    :return:
    """
    dict = {}
    dict['epoch'] = ep_ind
    dict['loss'] = best_loss
    np.save(config_path_var, dict)

