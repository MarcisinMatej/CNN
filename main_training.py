from CNN import *
from keras import optimizers
from data_proc.DataGenerator import DataGenerator
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

bulk_size = 2048
model_path = 'models/'
n_epochs = 250
batch_size = 64
in_shape = (64, 64, 3)


def train_epoch(model, generator, ep_ind):
    """
    Procedure to train provided model with provide data from generator
    in single epoch. After training results are plotted with plot_history(...) function.
    :param model:
    :param generator: yields through all training data
    :param ep_ind: index of epoch
    """
    histories_train = []
    train_gen = generator.generate_training()
    ep_hist_train = {}

    # training
    for X_train, Y_train in train_gen:  # these are chunks of ~bulk pictures
        # TODO here we can select just 1 attribute for training
        histories_train.append(model.fit(X_train, Y_train, batch_size=batch_size, epochs=1))

    save_model(model, model_path)
    plot_history(merge_history(histories_train), ep_hist_train, str(ep_ind) + 'epoch_train')


def validate_epoch(model, generator, epoch_id):
    """
    Procedure to validate provided model with provide data from generator
    in single epoch. After evaluation the result is plotted with plot_history(...) function.
    :param model:
    :param generator:
    :param epoch_id:
    :return:
    """
    hist_tst = []
    ep_hist_tst = {}
    val_gen = generator.generate_validation()
    for X_train, Y_train in val_gen:  # these are chunks of ~bulk pictures
        hist_tst.append(model.evaluate(x=X_train, y=Y_train, batch_size=batch_size))
    plot_history(prepare_eval_history(hist_tst), ep_hist_tst, str(epoch_id) + 'epoch_validation')


def run_model():
    """
    Prepares fresh new model and network and runs it.
    :return:
    """
    model = define_network(in_shape=in_shape)
    opt = optimizers.Adam(lr=0.0000015)
    model.compile(optimizer=opt,loss= "categorical_crossentropy",loss_weights=[1, 1, 1, 1, 1], metrics=['accuracy'])

    for e in range(n_epochs):
        print("epoch %d" % e)
        generator = DataGenerator((64, 64), bulk_size)
        train_epoch(model, generator, e)
        # Validing epoch
        validate_epoch(model, generator, e)


def run_load_model():
    """
    Loads model from saved location and runs it.
    :return:
    """
    model = load_model(model_path)
    opt = optimizers.Adam(lr=0.0000015)
    # model.compile(optimizer=rms, loss=["categorical_crossentropy", "categorical_crossentropy","categorical_crossentropy", "categorical_crossentropy","categorical_crossentropy"], metrics=['accuracy'])
    model.compile(optimizer=opt,loss= "categorical_crossentropy", metrics=['accuracy'])

    for e in range(n_epochs):
        print("epoch %d" % e)
        generator = DataGenerator((64, 64), bulk_size)
        # Training
        train_epoch(model, generator, e)
        # Validating
        validate_epoch(model, generator, e)


def run_model_virtual():
    """
    Model is freshly created and run with additional virtual examples.
    :return:
    """
    model = define_network(in_shape=in_shape)
    opt = optimizers.Adam(lr=0.0000015)
    # model.compile(optimizer=rms, loss=["categorical_crossentropy", "categorical_crossentropy","categorical_crossentropy", "categorical_crossentropy","categorical_crossentropy"], metrics=['accuracy'])
    model.compile(optimizer=opt, loss="categorical_crossentropy", loss_weights=[1, 1, 1, 1, 1], metrics=['accuracy'])

    histories_train = []
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
        # Validating
        validate_epoch(model, tmp, e)


if __name__ == "__main__":
    # issue with memory, in default tensorflow allocates nearly all possible memory
    # this can result in OOM error later
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    run_model()

    # RunLoadedModelWithGenerators()
