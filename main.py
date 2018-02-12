import keras

from CNN import *
from keras import optimizers
from data_proc.DataGenerator import DataGenerator
import tensorflow as tf


def RunModel(train_data,train_labels_one_hot,test_data,test_labels_one_hot):
    model = define_network()
    batch_size = 200
    epochs = 100
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, train_labels_one_hot, batch_size=batch_size, epochs=epochs, verbose=1,
                         validation_data=(test_data, test_labels_one_hot))

    # Plot training progress
    plot_history(history)

    # Score trained model.
    scores = model.evaluate(test_data, test_labels_one_hot)
    # scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

def RunModelWithGenerators():
    bulk_size = 32
    model_path = 'model/'
    n_epochs = 250
    batch_size = 32
    in_shape = (64,64,3)
    model = define_network(in_shape=in_shape)
    opt = optimizers.Adam(lr=0.0000015)
    # model.compile(optimizer=rms, loss=["categorical_crossentropy", "categorical_crossentropy","categorical_crossentropy", "categorical_crossentropy","categorical_crossentropy"], metrics=['accuracy'])
    model.compile(optimizer=opt,loss= "categorical_crossentropy",loss_weights=[1, 1, 1, 1, 1], metrics=['accuracy'])

    histories_train = []
    histories_test = []

    for e in range(n_epochs):
        print("epoch %d" % e)
        tmp = DataGenerator((64, 64), bulk_size)
        train_gen = tmp.generate_training()
        # training
        for X_train, Y_train in train_gen:  # these are chunks of ~bulk pictures
            #TODO here we can select just 1 attribute for training
            histories_train.append(model.fit(X_train, Y_train, batch_size=batch_size, epochs=1))

        save_model(model,model_path)
        plot_history(merge_history(histories_train), 'epoch_train' + str(e))
        # Testing
        test_gen = tmp.generate_testing()
        for X_train, Y_train in test_gen:  # these are chunks of ~bulk pictures
            histories_test.append(model.evaluate(x=X_train,y=Y_train,batch_size=batch_size))
        plot_history(prepare_eval_history(histories_test), 'epoch_test' + str(e))


def RunLoadedModelWithGenerators():
    bulk_size = 1024
    model_path = 'model/'
    n_epochs = 250
    batch_size = 32

    model = load_model(model_path)
    opt = optimizers.Adam(lr=0.0000015)
    # model.compile(optimizer=rms, loss=["categorical_crossentropy", "categorical_crossentropy","categorical_crossentropy", "categorical_crossentropy","categorical_crossentropy"], metrics=['accuracy'])
    model.compile(optimizer=opt,loss= "categorical_crossentropy", metrics=['accuracy'])

    histories_train = []
    histories_test = []

    for e in range(n_epochs):
        print("epoch %d" % e)
        tmp = DataGenerator((64, 64), bulk_size)
        train_gen = tmp.generate_training()
        # training
        for X_train, Y_train in train_gen:  # these are chunks of ~bulk pictures
            #TODO here we can select just 1 attribute for training
            histories_train.append(model.fit(X_train, Y_train, batch_size=batch_size, epochs=1))

        save_model(model,model_path)
        plot_history(merge_history(histories_train), 'epoch_train' + str(e))
        # Testing
        test_gen = tmp.generate_testing()
        c = 0
        for X_test, Y_test in test_gen:  # these are chunks of ~bulk pictures
            print("chunk" + str(c))
            c+=1
            histories_test.append(model.evaluate(x=X_test,y=Y_test,batch_size=batch_size))
        plot_history(prepare_eval_history(histories_test), 'epoch_test' + str(e))


def RunModelWithVirtualGenerators():
    bulk_size = 1024
    model_path = 'model/'
    n_epochs = 250
    batch_size = 32

    in_shape = (64, 64, 3)
    model = define_network(in_shape=in_shape)
    opt = optimizers.Adam(lr=0.0000015)
    # model.compile(optimizer=rms, loss=["categorical_crossentropy", "categorical_crossentropy","categorical_crossentropy", "categorical_crossentropy","categorical_crossentropy"], metrics=['accuracy'])
    model.compile(optimizer=opt, loss="categorical_crossentropy", loss_weights=[1, 1, 1, 1, 1], metrics=['accuracy'])

    histories_train = []
    histories_test = []

    from keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=True,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    #TODO what is representable sample???
    # datagen.fit(X_sample)  # let's say X_sample is a small-ish but statistically representative sample of your data

    # let's say you have an ImageNet generator that yields ~10k samples at a time.
    # for e in range(n_epochs):
    #     print("epoch %d" % e)
    #     for X_train, Y_train in ImageNet():  # these are chunks of ~10k pictures
    #         for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size=32):  # these are chunks of 32 samples
    #             loss = model.train(X_batch, Y_batch)

    for e in range(n_epochs):
        print("epoch %d" % e)
        tmp = DataGenerator((64, 64), bulk_size)
        train_gen = tmp.generate_training()
        # training
        for X_train, Y_train in train_gen:  # these are chunks of ~bulk pictures
            for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size=32):  # these are chunks of 32 samples
            #TODO here we can select just 1 attribute for training
                histories_train.append(model.fit(X_batch, Y_batch, batch_size=batch_size, epochs=1))

        save_model(model,model_path)
        plot_history(merge_history(histories_train), 'epoch_train' + str(e))
        # Testing
        test_gen = tmp.generate_testing()
        c = 0
        for X_test, Y_test in test_gen:  # these are chunks of ~bulk pictures
            print("chunk" + str(c))
            c+=1
            histories_test.append(model.evaluate(x=X_test,y=Y_test,batch_size=batch_size))
        plot_history(prepare_eval_history(histories_test), 'epoch_test' + str(e))





if __name__ == "__main__":
    # issue with memory, in default tensorflow allocates nearly all possible memory
    # this can result in OOM error later
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    RunModelWithGenerators()

    # RunLoadedModelWithGenerators()