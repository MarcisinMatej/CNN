from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense
from keras.layers import Input, Flatten

from keras.models import Model, model_from_json
from keras.utils import plot_model

from data_proc.DataLoader import get_attributes_desc

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
figures_path = 'figures/'

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
    _plot_model(model)
    return model


def _plot_model(model):
    """
    Produces plot of model, see keras.plot_model
    :param model:
    :return: saves plot image
    """
    plot_model(model, to_file='model.png')


def plot_loss(data, epoch_ind):
    """
    Plots loss curves from data dictionary.
    :param data: dictionary in form, where each key has loss in its name
     e.g. : 'string_loss':[list of integers]
    :param epoch_ind: index of epoch
    :return: saved plot as png file which name is starting by epoch_index
    """
    # Loss Curves
    c = 0
    plt.figure()
    ax = plt.subplot(111)
    for key in data.keys():
        if 'loss' in key:
            ax.plot(data[key], COLORS[c], linewidth=3.0, label=key)
            c = (c + 1) % len(COLORS)
            # plt.plot(history.history['loss'], 'r', linewidth=3.0)
            # plt.plot(history['loss'], 'r', linewidth=3.0)
            # plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
            # plt.plot(history['val_loss'], 'b', linewidth=3.0)

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    ax.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, -0.05))
    plt.xlabel('Chunks ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)
    plt.savefig(figures_path + epoch_ind + "_loss")
    plt.close('all')


def plot_accuracy(data, epoch_ind):
    """
        Plots accuracy curves from data dictionary.
        :param data: dictionary in form, where each key has 'acc' substring in its name
         e.g. : 'string_acc':[list of integers]
        :param epoch_ind: index of epoch
        :return: saved plot as png file which name is starting by epoch_index
        """
    # Accuracy Curves
    c = 0
    plt.figure(figsize=[8, 6])
    ax = plt.subplot(111)
    for key in data.keys():
        if 'acc' in key:
            ax.plot(data[key], COLORS[c], linewidth=3.0, label=key)
            c = (c + 1) % len(COLORS)
            # #plt.plot(history.history['acc'], 'r', linewidth=3.0)
            # plt.plot(history['acc'], 'r', linewidth=3.0)
            # #plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
            # plt.plot(history['val_acc'], 'b', linewidth=3.0)
            # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    ax.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, -0.05))
    plt.xlabel('Chunks ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)
    plt.savefig(figures_path + epoch_ind + '_acc')
    plt.close('all')


def plot_history(history, agg_history, epoch_ind,agg=True):
    """
    Produces plot of loss and accuracy per epoch for validation and training data.
    Values are taken from history.
    :param history: history returned from merge_histories(), basically dictionary of loss/accuracy.
    Contains values of metrics per bulk size
    :param agg_history: average metrics per epoch
    :param epoch_ind: index of epoch
    :return:
    """
    # plot metrics in the last epoch
    plot_loss(history, epoch_ind)
    plot_accuracy(history, epoch_ind)
    if agg:
        # plot avegare metrics thorugh all epochs
        agg_history = merge_epoch_history(agg_history,history)
        plot_loss(agg_history, "aggregate")
        plot_accuracy(agg_history, "aggregate")
    plt.close('all')


def merge_history(histories):
    """
    Helper method which merges multiple history files.
    :param histories: array of histories ordered in order of merging (needed for epoch index update)
    :return: single history dictionary
    """
    history = {}
    if len(histories) <= 0:
        return {}
    for key in histories[0].history.keys():
        history[key] = []

    for h in histories:
        for key in h.history.keys():
            history[key] += h.history[key]

    return history

def merge_epoch_history(previous, current):
    """
    Helper method which merges multiple dictionaries to average value for epoch.
    :param previous : old history dictionary per epoch
    :param current : new data from the last epoch
    :return: single history dictionary
    """
    if len(current.keys()) <= 0:
        return previous
    if len(previous.keys()) <= 0:
        for key in current.keys():
            previous[key] = []

    for key in current.keys():
        #average new
        previous[key].append(sum(current[key]) / len(current[key]))
    return previous

def prepare_eval_history(histories):
    history = {}
    if len(histories) <= 0:
        return history

    att_cnt = len(histories[0])
    # first is aggragate loss, the rest is split half loss and the second half acc
    mat = np.asarray(histories)
    history['Agg_loss'] = np.asarray(mat[:,0])

    for i in range(1, int(att_cnt/2) + 1):
        history['loss' + str(i)] = mat[:,i]
    for i in range(1, int(att_cnt/2) + 1):
        history['acc' + str(i)] = mat[:,int(att_cnt/2) + i]
    return history

def save_model(model,path):
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
    print("Saved model to disk")

def load_model(path):
    """
    Loads KERAS model from designated location. Model is loaded
    from json_file and weights are separately loaded from model.h5 file.
    :param path: path to folder with 'model.json' file and 'model.h5' files.
    :return: loaded model with weights
    """
    # load json and create model
    json_file = open(path+'model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path+"model.h5")
    print("Loaded model from disk")
    return loaded_model





