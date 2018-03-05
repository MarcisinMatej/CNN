from CNN import *
from keras import optimizers
from data_proc.DataGenerator import DataGenerator
import tensorflow as tf

from main_plots import plot_history, merge_history, prepare_eval_history

bulk_size = 10240
model_path = 'models/'
n_epochs = 100
batch_size = 124
in_shape = (64, 64, 3)
VIRT_GEN_STOP = 1
BEST_LOSS = 999999999
learning_rate = 0.0000013


def train_epoch(model, generator, ep_ind, ep_hist_train):
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

    save_model(model, model_path,ep_ind,BEST_LOSS)
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
    global BEST_LOSS
    hist_val = []
    val_gen = generator.generate_validation()
    agg = 0
    for X_train, Y_train in val_gen:  # these are chunks of ~bulk pictures
        hist_val.append(model.evaluate(x=X_train, y=Y_train, batch_size=batch_size))
        agg += hist_val[-1][0]
    plot_history(prepare_eval_history(hist_val), ep_hist_val, str(epoch_id) + 'epoch_validation')
    # save model if we get better validation loss

    if agg < BEST_LOSS:
        print("!!!!!!!!!!!!!!!!!!  AGG LOSS IMPROVEMENT, now:" + str(BEST_LOSS) + ", new:" + str(agg))
        save_model(model, model_path+"best_",epoch_id,BEST_LOSS)
        BEST_LOSS = agg

def run_model():
    """
    Prepares fresh new model and network and runs it.
    :return:
    """
    model = define_network(in_shape=in_shape)
    opt = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=opt,loss= "categorical_crossentropy",loss_weights=[1, 1, 1, 1, 1], metrics=['accuracy'])
    ep_hist_train = {}
    ep_hist_val = {}
    generator = DataGenerator((64, 64), bulk_size)
    for e in range(n_epochs):
        print("epoch %d" % e)
        train_epoch(model, generator, e, ep_hist_train)
        # Validing epoch
        validate_epoch(model, generator, e, ep_hist_val)


def run_load_model():
    """
    Loads model from saved location and runs it.
    :return:
    """
    global BEST_LOSS
    ep_hist_train = {}
    ep_hist_val = {}
    model,vars_dict = load_model(model_path)
    BEST_LOSS = vars_dict["loss"]
    start_ep = vars_dict["epoch"]
    opt = optimizers.Adam(lr=learning_rate)
    # model.compile(optimizer=rms, loss=["categorical_crossentropy", "categorical_crossentropy","categorical_crossentropy", "categorical_crossentropy","categorical_crossentropy"], metrics=['accuracy'])
    model.compile(optimizer=opt,loss= "categorical_crossentropy", metrics=['accuracy'])
    generator = DataGenerator((64, 64), bulk_size)

    for e in range(start_ep,n_epochs):
        print("epoch %d" % e)
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
    opt = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=opt, loss="categorical_crossentropy", loss_weights=[1, 1, 1, 1, 1], metrics=['accuracy'])
    ep_hist_train = {}
    ep_hist_val = {}
    generator = DataGenerator((64, 64), bulk_size)
    for e in range(n_epochs):
        print("epoch %d" % e)
        histories_train = []
        train_gen = generator.virtual_train_generator()

        # training
        for X_train, Y_train in train_gen:  # these are chunks of ~bulk pictures
            # TODO here we can select just 1 attribute for training
            histories_train.append(model.fit(X_train, Y_train, batch_size=batch_size, epochs=1))

        save_model(model, model_path,e,BEST_LOSS)
        plot_history(merge_history(histories_train), ep_hist_train, str(e) + 'epoch_train')
        # Validing epoch
        validate_epoch(model, generator, e, ep_hist_val)


if __name__ == "__main__":
    # issue with memory, in default tensorflow allocates nearly all possible memory
    # this can result in OOM error later
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    run_model()
    # run_model_virtual()
    # RunLoadedModelWithGenerators()
    # path="histories/0epoch_train_hist.npy"
    # print(load_history(path))
    # run_load_model()