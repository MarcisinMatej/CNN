from __future__ import print_function

import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation
from keras.layers import Input, Flatten

from keras.models import Model
from keras.utils import plot_model


def createModel(in_shape=(32, 32, 3)):
    #TODO add parameter for number of outputs and type categories for softmax
    '''
    Creates model for CNN with Keras functional api.
    :param in_shape: tuple (x,y,z) of input shape for model
    :return: keras model
    '''

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

    #TODO
    # update to dynamic values depending on attribute file
    # output layers
    flatten = Flatten()(conv8)
    output_layers = []
    out1 = Dense(2, activation='softmax',name="out1")(flatten)
    output_layers.append(out1)
    out2 = Dense(2, activation='softmax',name="out2")(flatten)
    output_layers.append(out2)
    out3 = Dense(2, activation='softmax',name="out3")(flatten)
    output_layers.append(out3)
    out4 = Dense(2, activation='softmax',name="out4")(flatten)
    output_layers.append(out4)
    out5 = Dense(5, activation='softmax',name="out5")(flatten)
    output_layers.append(out5)

    #TODO here we must also change to single output
    model = Model(inputs=input, outputs=output_layers)
    # summarize layers
    print(model.summary())
    return model


def PlotModel(model):
    #TODO fix graphviz
    '''
    Produces plot of model
    :param model:
    :return: saves plot image
    '''
    plot_model(model, to_file='model.png')


def PlotHistory(history):
    '''
    Produces plot of loss and accuracy per epoch for validation and training data.
    Values are taken from history.
    :param history: history returned from fit.model(...), basically dictionary
    :return:
    '''
    # Loss Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['loss'], 'r', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)

    # Accuracy Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['acc'], 'r', linewidth=3.0)
    plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)

def MergeHistory(histories):
    '''
    Helper method which merges multiple history files.
    :param histories: array of histories ordered in order of merging (needed for epoch index update)
    :return: single history dictionary
    '''
    pass
    #TODO






