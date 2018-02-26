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
VIRT_GEN_STOP = 1


def train_epoch(model, generator, ep_ind,ep_hist_train):
    """
    Procedure to train provided model with provide data from generator
    in single epoch. After training results are plotted with plot_history(...) function.
    :param model:
    :param generator: yields through all training data
    :param ep_ind: index of epoch
    """
    histories_train = []
    train_gen = generator.generate_training()

    # training
    for X_train, Y_train in train_gen:  # these are chunks of ~bulk pictures
        # TODO here we can select just 1 attribute for training
        histories_train.append(model.fit(X_train, Y_train, batch_size=batch_size, epochs=1))

    save_model(model, model_path)
    plot_history(merge_history(histories_train), ep_hist_train, str(ep_ind) + 'epoch_train')


def validate_epoch(model, generator, epoch_id,ep_hist_val):
    """
    Procedure to validate provided model with provide data from generator
    in single epoch. After evaluation the result is plotted with plot_history(...) function.
    :param model:
    :param generator:
    :param epoch_id:
    :return:
    """
    hist_val = []
    val_gen = generator.generate_validation()
    for X_train, Y_train in val_gen:  # these are chunks of ~bulk pictures
        hist_val.append(model.evaluate(x=X_train, y=Y_train, batch_size=batch_size))
    plot_history(prepare_eval_history(hist_val), ep_hist_val, str(epoch_id) + 'epoch_validation')


def run_model():
    """
    Prepares fresh new model and network and runs it.
    :return:
    """
    model = define_network(in_shape=in_shape)
    opt = optimizers.Adam(lr=0.0000015)
    model.compile(optimizer=opt,loss= "categorical_crossentropy",loss_weights=[1, 1, 1, 1, 1], metrics=['accuracy'])
    ep_hist_train = {}
    ep_hist_val = {}
    for e in range(n_epochs):
        print("epoch %d" % e)
        generator = DataGenerator((64, 64), bulk_size)
        train_epoch(model, generator, e, ep_hist_train)
        # Validing epoch
        validate_epoch(model, generator, e, ep_hist_val)


def run_load_model():
    """
    Loads model from saved location and runs it.
    :return:
    """
    ep_hist_train,ep_hist_val, model = load_model(model_path)
    opt = optimizers.Adam(lr=0.0000015)
    # model.compile(optimizer=rms, loss=["categorical_crossentropy", "categorical_crossentropy","categorical_crossentropy", "categorical_crossentropy","categorical_crossentropy"], metrics=['accuracy'])
    model.compile(optimizer=opt,loss= "categorical_crossentropy", metrics=['accuracy'])

    for e in range(n_epochs):
        print("epoch %d" % e)
        generator = DataGenerator((64, 64), bulk_size)
        # Training
        train_epoch(model, generator, e, ep_hist_train)
        # Validating
        validate_epoch(model, generator, e, ep_hist_val)


def run_model_virtual():
    """
    Model is freshly created and run with additional virtual examples.
    :return:
    """
    model = define_network(in_shape=in_shape)
    opt = optimizers.Adam(lr=0.0000015)
    model.compile(optimizer=opt, loss="categorical_crossentropy", loss_weights=[1, 1, 1, 1, 1], metrics=['accuracy'])
    ep_hist_train = {}
    ep_hist_val = {}
    for e in range(n_epochs):
        print("epoch %d" % e)
        generator = DataGenerator((64, 64), bulk_size)
        histories_train = []
        train_gen = generator.virtual_train_generator()

        # training
        for X_train, Y_train in train_gen:  # these are chunks of ~bulk pictures
            # TODO here we can select just 1 attribute for training
            histories_train.append(model.fit(X_train, Y_train, batch_size=batch_size, epochs=1))

        save_model(model, model_path)
        plot_history(merge_history(histories_train), ep_hist_train, str(e) + 'epoch_train')
        # Validing epoch
        validate_epoch(model, generator, e, ep_hist_val)


if __name__ == "__main__":
    # issue with memory, in default tensorflow allocates nearly all possible memory
    # this can result in OOM error later
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # run_model()
    run_model_virtual()
    # RunLoadedModelWithGenerators()
