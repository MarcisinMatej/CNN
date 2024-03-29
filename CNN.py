from __future__ import print_function
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout, ZeroPadding2D
from keras.layers import Dense
from keras.layers import Input, Flatten
from keras.models import Model, model_from_json
from keras.utils import plot_model

from data_proc.ConfigLoaderCelebA import get_attributes_desc, get_category_names
# from main_plots import _plot_model

history_path = "histories/"
config_path_var = "data_proc/config_files/training_vars.npy"


def Conv2DBatchNormRelu(n_filter, w_filter, h_filter, inputs, ind,_padding="valid"):
    return Activation(activation='relu',name="ReLU_"+ind)(BatchNormalization(name="BN_"+ind)
                    (Conv2D(n_filter, (w_filter, h_filter), padding=_padding,name="Conv_"+str(n_filter)+"_"+ind)(inputs)))


def define_network_2_with_BN(in_shape=(32, 32, 3), single_output_ind = None):
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
    maxp1 = MaxPooling2D(pool_size=(2, 2), strides=2, data_format=channel, name="MP_1")(conv2)
    conv3 = Conv2DBatchNormRelu(64, 3, 3, inputs=maxp1,ind='3')
    maxp2 = MaxPooling2D(pool_size=(2, 2), strides=2, data_format=channel, name="MP_2")(conv3)
    conv4 = Conv2DBatchNormRelu(64, 3, 3, inputs=maxp2,ind='4')
    maxp3 = MaxPooling2D(pool_size=(2, 2), strides=2, data_format=channel, name="MP_3")(conv4)
    conv5 = Conv2DBatchNormRelu(128, 3, 3, inputs=maxp3,ind='5')
    conv6 = Conv2DBatchNormRelu(128, 4, 4, inputs=conv5,ind='6')
    conv7 = Conv2DBatchNormRelu(2048, 5, 5, inputs=conv6,ind='7')
    output_layers = []

    output_layers = []
    # output layers
    conv8 = Conv2D(2048, (1, 1), name='Conv_8')(conv7)
    flatten1 = Flatten()(conv8)
    out1 = Dense(2, activation='softmax',name="Attractiveness")(flatten1)
    output_layers.append(out1)

    conv9 = Conv2D(2048, (1, 1), name='Conv_9')(conv7)
    flatten2 = Flatten()(conv9)
    out2 = Dense(2, activation='softmax',name="Glass")(flatten2)
    output_layers.append(out2)
    out3 = Dense(2, activation='softmax',name="Gender")(flatten2)
    output_layers.append(out3)
    out4 = Dense(2, activation='softmax',name="Smile")(flatten2)
    output_layers.append(out4)

    conv12 = Conv2DBatchNormRelu(128, 3, 3, inputs=maxp3, ind='12',_padding="same")
    conv13 = Conv2DBatchNormRelu(128, 4, 4, inputs=conv12, ind='13')
    conv14 = Conv2DBatchNormRelu(2048, 5, 5, inputs=conv13, ind='14')
    conv15 = Conv2D(2048, (1, 1), name='Conv_15')(conv14)
    flatten5 = Flatten()(conv15)
    out5 = Dense(5, activation='softmax',name="Hair")(flatten5)
    output_layers.append(out5)

    model = Model(inputs=inputs, outputs=output_layers)


    # summarize layers
    print(model.summary())
    #TODO plot model
    # _plot_model(model)
    # plot_model(model, to_file='model.png')
    return model


def define_network_with_BN(in_shape=(32, 32, 3), single_output_ind=None):
    """
    Creates model for CNN with Keras functional api.
    Model graph is fixed and before each activation function is
    normalization layer. The output layers are defined by
    'attributes_values.txt' file. The number of lines is number of outputs
    and number of categories is determined by number of listed categories for each
    attribute.
    :param single_output_ind: index of desired single output. If not required leave None.
    :param in_shape: tuple (x,y,z) of input shape for model
    :return: keras model
    """

    channel = "channels_last"
    #define shape of input
    inputs = Input(shape=in_shape,name="input")

    # define layers relationships
    conv1 = Conv2DBatchNormRelu(32, 3, 3, inputs=inputs,ind='1')
    conv2 = Conv2DBatchNormRelu(32,3, 3, inputs=conv1,ind='2')
    maxp1 = MaxPooling2D(pool_size=(2, 2), strides=2, data_format=channel, name="MP_1")(conv2)
    conv3 = Conv2DBatchNormRelu(64, 3, 3, inputs=maxp1,ind='3')
    maxp2 = MaxPooling2D(pool_size=(2, 2), strides=2, data_format=channel, name="MP_2")(conv3)
    conv4 = Conv2DBatchNormRelu(64, 3, 3, inputs=maxp2,ind='4')
    maxp3 = MaxPooling2D(pool_size=(2, 2), strides=2, data_format=channel, name="MP_3")(conv4)
    conv5 = Conv2DBatchNormRelu(128, 3, 3, inputs=maxp3,ind='5')
    conv6 = Conv2DBatchNormRelu(128, 4, 4, inputs=conv5,ind='6')
    conv7 = Conv2DBatchNormRelu(2048, 5, 5, inputs=conv6,ind='7')
    # conv8 = Conv2DBatchNormRelu(2048, 1, 1, inputs=conv7,ind='8')
    conv8 = Conv2D(2048, (1, 1), name='Conv_8')(conv7)
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
    # plot_model(model, to_file='model.png')
    return model


def define_network(in_shape=(32, 32, 3), single_output_ind=None):
    """
    Creates model for CNN with Keras functional api.
    Model graph is fixed. The output layers are defined by
    'attributes_values.txt' file. The number of lines is number of outputs
    and number of categories is determined by number of listed categories for each
    attribute.
    :param single_output_ind: index of desired single output. If not required leave None.
    :param in_shape: tuple (x,y,z) of input shape for model
    :return: keras model
    """

    channel = "channels_last"
    act = "relu"
    padd = "valid"
    #define shape of input
    inputs = Input(shape=in_shape, name="input")

    # define layers relationships
    conv1 = Conv2D(32, (3, 3), padding=padd, activation=act)(inputs)
    conv2 = Conv2D(32, (3, 3), padding=padd, activation=act)(conv1)
    maxp1 = MaxPooling2D(pool_size=(2, 2), strides=2,data_format=channel)(conv2)
    conv3 = Conv2D(64, (3, 3), padding=padd, activation=act)(maxp1)
    maxp2 = MaxPooling2D(pool_size=(2, 2), strides=2,data_format=channel)(conv3)
    conv4 = Conv2D(64, (3, 3), padding=padd, activation=act)(maxp2)
    maxp3 = MaxPooling2D(pool_size=(2, 2), strides=2,data_format=channel)(conv4)
    conv5 = Conv2D(128, (3, 3), padding=padd, activation=act)(maxp3)
    conv6 = Conv2D(128, (4, 4), padding=padd, activation=act)(conv5)
    conv7 = Conv2D(2048, (5, 5), padding=padd, activation=act)(conv6)
    conv8 = Conv2D(2048, (1, 1), padding=padd, activation=act)(conv7)
    flatten = Flatten()(conv8)
    output_layers = []

    atrs_desc = get_attributes_desc()
    cat_names = get_category_names()
    for cnt, _name in zip(atrs_desc, cat_names):
        output_layers.append(Dense(cnt, activation='softmax', name=_name)(flatten))

    if single_output_ind is None:
        model = Model(inputs=inputs, outputs=output_layers)
    else:
        model = Model(inputs=inputs, outputs=output_layers[single_output_ind])
    # summarize layers
    print(model.summary())
    plot_model(model, to_file='model.png')
    return model


def define_network_multi(out_arity, out_names, in_shape=(32, 32, 3)):
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
    act = "relu"
    padd = "valid"
    # define shape of input
    inputs = Input(shape=in_shape, name="input")

    # define layers relationships
    conv1 = Conv2D(32, (3, 3), padding=padd, activation=act)(inputs)
    conv2 = Conv2D(32, (3, 3), padding=padd, activation=act)(conv1)
    maxp1 = MaxPooling2D(pool_size=(2, 2), strides=2, data_format=channel)(conv2)
    conv3 = Conv2D(64, (3, 3), padding=padd, activation=act)(maxp1)
    maxp2 = MaxPooling2D(pool_size=(2, 2), strides=2, data_format=channel)(conv3)
    conv4 = Conv2D(64, (3, 3), padding=padd, activation=act)(maxp2)
    maxp3 = MaxPooling2D(pool_size=(2, 2), strides=2, data_format=channel)(conv4)
    conv5 = Conv2D(128, (3, 3), padding=padd, activation=act)(maxp3)
    conv6 = Conv2D(128, (4, 4), padding=padd, activation=act)(conv5)
    conv7 = Conv2D(2048, (5, 5), padding=padd, activation=act)(conv6)
    conv8 = Conv2D(2048, (1, 1), padding=padd, activation=act)(conv7)
    conv9 = Conv2D(4096, (1, 1), padding=padd)(conv8)
    flatten = Flatten()(conv9)
    output_layers = []

    for cnt, _name in zip(out_arity, out_names):
        output_layers.append(Dense(cnt, activation='softmax', name=_name)(flatten))
    model = Model(inputs=inputs, outputs=output_layers)

    # summarize layers
    print(model.summary())
    #TODO plot model
    # _plot_model(model)
    # plot_model(model, to_file='model.png')
    return model


def define_network_BN_multi(out_arity, out_names, in_shape=(32, 32, 3)):
    """
    Creates model for CNN with Keras functional api.
    Model graph is fixed. The output layers are defined by
    'attributes_values.txt' file. The number of lines is number of outputs
    and number of categories is determined by number of listed categories for each
    attribute.
    :param out_arity: array of output dimensions eg [2,3] means first output
    has 2 categorical values and the second output has 3 categorical values.
    :param out_names: list of string names of outputs
    :param in_shape: tuple (x,y,z) of input shape for model
    :return: keras model
    """

    channel = "channels_last"
    #define shape of input
    inputs = Input(shape=in_shape,name="input")

    # define layers relationships
    conv1 = Conv2DBatchNormRelu(32, 3, 3, inputs=inputs, ind='1')
    conv2 = Conv2DBatchNormRelu(32, 3, 3, inputs=conv1, ind='2')
    maxp1 = MaxPooling2D(pool_size=(2, 2), strides=2, data_format=channel, name="MP_1")(conv2)
    conv3 = Conv2DBatchNormRelu(64, 3, 3, inputs=maxp1, ind='3')
    maxp2 = MaxPooling2D(pool_size=(2, 2), strides=2, data_format=channel, name="MP_2")(conv3)
    conv4 = Conv2DBatchNormRelu(64, 3, 3, inputs=maxp2, ind='4')
    maxp3 = MaxPooling2D(pool_size=(2, 2), strides=2, data_format=channel, name="MP_3")(conv4)
    conv5 = Conv2DBatchNormRelu(128, 3, 3, inputs=maxp3, ind='5')
    conv6 = Conv2DBatchNormRelu(128, 4, 4, inputs=conv5, ind='6')
    conv7 = Conv2DBatchNormRelu(2048, 5, 5, inputs=conv6, ind='7')
    conv8 = Conv2D(2048, (1, 1), name='Conv_8')(conv7)
    flatten = Flatten()(conv8)
    output_layers = []

    for cnt, _name in zip(out_arity, out_names):
        output_layers.append(Dense(cnt, activation='softmax', name=_name)(flatten))
    model = Model(inputs=inputs, outputs=output_layers)

    # summarize layers
    print(model.summary())
    # _plot_model(model)
    plot_model(model, to_file='model.png')
    return model


def VGG_16(out_arity, out_names, in_shape=(32, 32, 3)):
    # model = Sequential()
    inputs = Input(shape=in_shape, name="input")
    zp1 = ZeroPadding2D((1, 1), input_shape=(3, 224, 224))(inputs)
    conv1 = Conv2D(64, 3, 3, activation='relu')(zp1)
    zp2 = ZeroPadding2D((1,1))(conv1)
    conv2 = Conv2D(64, 3, 3, activation='relu')(zp2)
    mp1 = MaxPooling2D((2,2), strides=(2,2))(conv2)

    zp3 = ZeroPadding2D((1,1))(mp1)
    conv3 = Conv2D(128, 3, 3, activation='relu')(zp3)
    zp4 = ZeroPadding2D((1,1))(conv3)
    conv4 = Conv2D(128, 3, 3, activation='relu')(zp4)
    mp2 = MaxPooling2D((2,2), strides=(2,2))(conv4)

    zp5 = ZeroPadding2D((1,1))(mp2)
    conv5 = Conv2D(256, 3, 3, activation='relu')(zp5)
    zp6 = ZeroPadding2D((1,1))(conv5)
    conv6 = Conv2D(256, 3, 3, activation='relu')(zp6)
    zp7 = ZeroPadding2D((1,1))(conv6)
    conv7 = Conv2D(256, 3, 3, activation='relu')(zp7)
    mp3 = MaxPooling2D((2,2), strides=(2,2))(conv7)

    zp8 = ZeroPadding2D((1,1))(mp3)
    conv8 = Conv2D(512, 3, 3, activation='relu')(zp8)
    zp9 = ZeroPadding2D((1,1))(conv8)
    conv9 = Conv2D(512, 3, 3, activation='relu')(zp9)
    zp10 = ZeroPadding2D((1,1))(conv9)
    conv10 = Conv2D(512, 3, 3, activation='relu')(zp10)
    mp4 = MaxPooling2D((2,2), strides=(2,2))(conv10)

    zp11 = ZeroPadding2D((1,1))(mp4)
    conv11 = Conv2D(512, 3, 3, activation='relu')(zp11)
    zp12 = ZeroPadding2D((1,1))(conv11)
    conv12 = Conv2D(512, 3, 3, activation='relu')(zp12)
    zp13 = ZeroPadding2D((1,1))(conv12)
    conv13 = Conv2D(512, 3, 3, activation='relu')(zp13)
    mp5 = MaxPooling2D((2,2), strides=(2,2))(conv13)

    flatten = Flatten()(mp5)
    dense1 = Dense(4096, activation='relu')(flatten)
    do1 = Dropout(0.5)(dense1)
    dense2 = Dense(4096, activation='relu')(do1)
    do2 = Dropout(0.5)(dense2)

    output_layers = []
    for cnt, _name in zip(out_arity, out_names):
        output_layers.append(Dense(cnt, activation='softmax', name=_name)(do2))
    model = Model(inputs=inputs, outputs=output_layers)

    # summarize layers
    print(model.summary())

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

def load_model_specific(path):
    """
    Loads KERAS model from designated location. Model is loaded
    from json_file and weights are separately loaded from model.h5 file.
    Location of config dictionary is config files directory.
    :param path: path to folder with 'model.json' file and 'model.h5' files.
    :return: loaded model with weights,dictionary of saved config variables (eg.epoch index, best validation loss)
    For more see save_vars()
    """
    # load json and create model
    json_file = open(path+".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path+".h5")
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
