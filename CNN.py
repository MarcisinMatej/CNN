from __future__ import print_function

import os

import keras
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation
from keras.layers import Input, Flatten
from keras import optimizers
from keras.models import Model
from keras.models import Sequential
from keras.utils import plot_model

from data_proc.data_loader import DataGenerator


def PrepareData():
    #TODO
    pass


def createModel(in_shape=(32, 32, 3)):
    '''

    :param in_shape: tuple (x,y,z) of input shape for model
    :return: keras model
    '''

    #define shape of input
    input = Input(shape=in_shape,name="input")

    # define layers relationships
    conv1 = Conv2D(32, (3, 3), padding='same',activation='relu')(input)
    conv2 = Conv2D(32, (3, 3),activation='relu')(conv1)
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
    plot_model(model, to_file='model.png')

def RunModel(train_data,train_labels_one_hot,test_data,test_labels_one_hot):
    model = createModel()
    batch_size = 200
    epochs = 100
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, train_labels_one_hot, batch_size=batch_size, epochs=epochs, verbose=1,
                         validation_data=(test_data, test_labels_one_hot))

    # Plot training progress
    PlotCurves(history)

    # Score trained model.
    scores = model.evaluate(test_data, test_labels_one_hot)
    # scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

def RunModelWithGenerators():
    train_data_dir = 'data/train'
    validation_data_dir = 'data/validation'
    nb_train_samples = 2000
    nb_validation_samples = 800
    epochs = 50
    batch_size = 16
    model = createModel()

    # model.compile(optimizer='rmsprop', loss=['categorical_crossentropy', 'categorical_crossentropy'],
    #               loss_weights=[1, 1], metrics=['accuracy'])
#"categorical_crossentropy", "categorical_crossentropy","categorical_crossentropy", "categorical_crossentropy","categorical_crossentropy"
    opt = optimizers.Adam(lr=0.0001)
    # model.compile(optimizer=rms, loss=["categorical_crossentropy", "categorical_crossentropy","categorical_crossentropy", "categorical_crossentropy","categorical_crossentropy"], metrics=['accuracy'])
    model.compile(optimizer=opt,loss= "categorical_crossentropy", metrics=['accuracy'])

    # model.fit(inputData, [outputYLeft, outputYRight], epochs=..., batch_size=...)

    tmp = DataGenerator((32,32),1000)
    gen = tmp.TrainingGenerator()

    histories = []
    for e in range(epochs):
        print("epoch %d" % e)
        for X_train, Y_train in gen:  # these are chunks of ~100 pictures
            print(X_train.shape)
            #TODO here we can select just 1 attribute for training
            histories.append(model.fit(X_train, Y_train, batch_size=32, epochs=1))


def PlotCurves(history):
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



RunModelWithGenerators()




