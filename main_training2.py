"""
Central script for running training of the CNN.
Available trainig scenarios:
    - Full run on dataset
        - CelebA
        - Imdb
        - Wiki
        - Adience
        - CelebA with partial annotation
        - CelebA+Imdb merged
        - CelebA+Wiki merged
    - Training with virtualization
    - Load model and continue learning
    - Load model with transfer knowledge
"""

import keras
import tensorflow as tf
from keras import optimizers
from keras.engine import Model
from keras.layers import Flatten, Dense

from CNN import save_model, load_model, define_network_with_BN, define_network, \
    define_network_multi, define_network_BN_multi, load_model_specific
from data_proc.DataGeneratorOUI import DataGeneratorOUI
from data_proc.DataGeneratorIMDB import DataGeneratorIMDB
from data_proc.DataGeneratorMerged import DataGeneratorMerged
from data_proc.DataGeneratorCelebA import DataGeneratorCelebA
from data_proc.DataGeneratorCelebASparse import DataGeneratorCelebASparse
from data_proc.DataGeneratorWiki import DataGeneratorWiki
from data_proc.DataGeneratorCelebAVirtual import DataGeneratorCelebAVirtual
from data_proc.VideoGenerator import VideoGenerator
from main_plots import plot_history, merge_history, prepare_eval_history
from keras import backend as K

"""
Main hyper parameters setup
"""
bulk_size = 12400
model_path = 'models/'
n_epochs = 100
batch_size = 124
in_shape = (100, 100, 3)
resolution = (100, 100)
VIRT_GEN_STOP = 1
# without BN
LEARNING_RATE = 0.000000314
# with BN
# learning_rate = 0.0000007

# support variables
BEST_LOSS = 999999999
BEST_EPOCH_IND = 0
MASK_VALUE = -1
VERBOSE = 1

def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.

    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.

    # Returns
        A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x

def masked_accuracy(target, output):
    """
    Accuracy metric for KERAS with partial labeling
    :param target: ground thruth values (MASK_VALUE signals missing label)
    :param output: predictions
    :return:
    """
    dtype = K.floatx()
    total = K.sum(K.cast(K.not_equal(target, MASK_VALUE), dtype))
    correct = K.sum(K.cast(K.equal(target, K.round(output)), dtype))
    return correct / total


def masked_loss_function(target, output):
    """
    Categorical cross entropy which is computed only for available labels.
    :param target: ground truth
    :param output: predictions
    :return:
    """
    mask = K.cast(K.not_equal(target, MASK_VALUE), K.floatx())
    return K.sparse_categorical_crossentropy(tf.multiply(target, mask), tf.multiply(output, mask))


def train_epoch(model, generator, ep_ind, ep_hist_train):
    """
    Procedure to train provided model with provide data from generator
    in single epoch. After training results are plotted with plot_history(...) function.
    :param model: Keras model to be trained.
    :param generator: yields through all training data
    :param ep_ind: index of epoch
    """
    histories_train = []
    train_gen = generator.generate_training()

    # training
    for X_train, Y_train in train_gen:  # these are chunks of ~bulk pictures
        # TODO here we can select just 1 attribute for training
        histories_train.append(model.fit(X_train, Y_train, batch_size=batch_size, epochs=1, verbose=VERBOSE))

    save_model(model, model_path,ep_ind,BEST_LOSS,BEST_EPOCH_IND)
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
    global BEST_LOSS,BEST_EPOCH_IND
    hist_val = []
    val_gen = generator.generate_validation()
    agg = 0
    for X_train, Y_train in val_gen:  # these are chunks of ~bulk pictures
        hist_val.append(model.evaluate(x=X_train, y=Y_train, batch_size=batch_size, verbose=VERBOSE))
        agg += hist_val[-1][0]
    plot_history(prepare_eval_history(hist_val), ep_hist_val, str(epoch_id) + 'epoch_validation')
    # save model if we get better validation loss

    if agg < BEST_LOSS:
        print("!!!!!!!!!!!!!!!!!!  AGG LOSS IMPROVEMENT, now:" + str(BEST_LOSS) + ", new:" + str(agg))
        save_model(model, model_path+"best_",epoch_id,BEST_LOSS,BEST_EPOCH_IND)
        BEST_LOSS = agg
        BEST_EPOCH_IND = epoch_id
        print(hist_val[-1])
    else:
        print("AGG LOSS NOT IMPROVED, BEST:" + str(BEST_LOSS) + ", current:" + str(agg))


def train_epoch_single(model, generator, ep_ind, ep_hist_train, out_index):
    """
    Procedure to train provided model with provide data from generator
    for single output scenario. After training results are plotted with plot_history(...) function.
    :param out_index: index of single output label from labels set
    :param ep_hist_train:
    :param model:
    :param generator: yields through all training data
    :param ep_ind: index of epoch
    """
    histories_train = []
    train_gen = generator.generate_training()

    # training
    for X_train, Y_train in train_gen:  # these are chunks of ~bulk pictures
        histories_train.append(model.fit(X_train, Y_train[out_index], batch_size=batch_size, epochs=1, verbose=VERBOSE))

    save_model(model, model_path,ep_ind,BEST_LOSS,BEST_EPOCH_IND)
    plot_history(merge_history(histories_train), ep_hist_train, str(ep_ind) + 'epoch_train')


def validate_epoch_single(model, generator, epoch_id, ep_hist_val, out_index):
    """
    Procedure to validate provided model with provide data from generator
    in single epoch. After evaluation the result is plotted with plot_history(...) function.
    :param model:
    :param generator:
    :param epoch_id:
    :return:
    """
    global BEST_LOSS,BEST_EPOCH_IND
    hist_val = []
    val_gen = generator.generate_validation()
    agg = 0
    for X_train, Y_train in val_gen:  # these are chunks of ~bulk pictures
        hist_val.append(model.evaluate(x=X_train, y=Y_train[out_index], batch_size=batch_size, verbose=VERBOSE))
        agg += hist_val[-1][0]
    plot_history(prepare_eval_history(hist_val), ep_hist_val, str(epoch_id) + 'epoch_validation')
    # save model if we get better validation loss

    if agg < BEST_LOSS:
        print("!!!!!!!!!!!!!!!!!!  AGG LOSS IMPROVEMENT, now:" + str(BEST_LOSS) + ", new:" + str(agg))
        save_model(model, model_path+"best_", epoch_id, BEST_LOSS, BEST_EPOCH_IND)
        BEST_LOSS = agg
        BEST_EPOCH_IND = epoch_id


def run_model_basic():
    """
    Prepares fresh new model and network and runs it.
    :return:
    """
    model = define_network(in_shape=in_shape)
    opt = optimizers.Adam(lr=LEARNING_RATE)
    model.compile(optimizer=opt,loss= "sparse_categorical_crossentropy",loss_weights=[1, 1, 1, 1, 1], metrics=['accuracy'])

    ep_hist_train = {}
    ep_hist_val = {}
    generator = DataGeneratorCelebA(resolution, bulk_size)
    for e in range(n_epochs):
        print("epoch %d" % e)
        train_epoch(model, generator, e, ep_hist_train)
        # Validing epoch
        validate_epoch(model, generator, e, ep_hist_val)


def run_model_imdb():
    """
    Prepares fresh new model for imdb dataset
    and trains it.
    :return:
    """
    model = define_network_multi([2, 6], ["Gender", "Age"], in_shape=in_shape)
    opt = optimizers.Adam(lr=LEARNING_RATE)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    ep_hist_train = {}
    ep_hist_val = {}
    generator = DataGeneratorIMDB(resolution, bulk_size)
    for e in range(n_epochs):
        print("epoch %d" % e)
        train_epoch(model, generator, e, ep_hist_train)
        validate_epoch(model, generator, e, ep_hist_val)



def run_model_adience():
    """
    Prepares fresh new model for Adience dataset
    and trains it.
    :return:
    """
    model = define_network_multi([2, 8], ["Gender", "Age"], in_shape=in_shape)
    opt = optimizers.Adam(lr=LEARNING_RATE)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    ep_hist_train = {}
    ep_hist_val = {}
    generator = DataGeneratorOUI(resolution, bulk_size)
    for e in range(n_epochs):
        print("epoch %d" % e)
        train_epoch(model, generator, e, ep_hist_train)
        validate_epoch(model, generator, e, ep_hist_val)


def run_model_wiki():
    """
    Prepares fresh new model for Wiki dataset trains it.
    :return:
    """
    model = define_network_multi([2,5],["Gender","Age"],in_shape=in_shape)
    opt = optimizers.Adam(lr=LEARNING_RATE)
    model.compile(optimizer=opt,loss= "sparse_categorical_crossentropy", metrics=['accuracy','mae'])

    ep_hist_train = {}
    ep_hist_val = {}
    generator = DataGeneratorWiki(resolution, bulk_size)
    for e in range(n_epochs):
        print("epoch %d" % e)
        train_epoch(model, generator, e, ep_hist_train)
        # Validing epoch
        validate_epoch(model, generator, e, ep_hist_val)


def run_model_merged():
    """
    Prepares fresh new model for Imdb + CelebA dataset.
    :return:
    """
    model = define_network_multi([2,2,2,2,5,6],["Attract","Glass","Gender","Smile","Hair","Age"],in_shape=in_shape)
    opt = optimizers.Adam(lr=LEARNING_RATE)
    model.compile(optimizer=opt, loss=masked_loss_function, metrics=['accuracy'])

    ep_hist_train = {}
    ep_hist_val = {}
    generator = DataGeneratorMerged(resolution, bulk_size)
    for e in range(n_epochs):
        print("epoch %d" % e)
        train_epoch(model, generator, e, ep_hist_train)
        # Validing epoch
        validate_epoch(model, generator, e, ep_hist_val)


def run_model_hidden():
    """
    Runs model for CelebA with partial labeling.
    :return:
    """
    model = define_network_with_BN(in_shape=in_shape)
    opt = optimizers.Adam(lr=LEARNING_RATE)
    # model.compile(optimizer=opt,loss= "sparse_categorical_crossentropy",loss_weights=[1, 1, 1, 1, 1], metrics=['accuracy'])
    model.compile(optimizer=opt, loss=masked_loss_function, loss_weights=[1, 1, 1, 1, 1],
                  metrics=['accuracy'])
    ep_hist_train = {}
    ep_hist_val = {}
    generator = DataGeneratorCelebASparse(resolution, bulk_size)
    for e in range(n_epochs):
        print("epoch %d" % e)
        train_epoch(model, generator, e, ep_hist_train)
        # Validing epoch
        validate_epoch(model, generator, e, ep_hist_val)

def run_model_with_single_out(ind):
    """
    Runs single output model from CelebA dataset.
    :return:
    """
    model = define_network(in_shape=in_shape, single_output_ind=ind)
    opt = optimizers.Adam(lr=LEARNING_RATE)
    model.compile(optimizer=opt,loss= "sparse_categorical_crossentropy", metrics=['accuracy'])
    ep_hist_train = {}
    ep_hist_val = {}
    generator = DataGeneratorCelebA(resolution, bulk_size)
    for e in range(n_epochs):
        print("epoch %d" % e)
        train_epoch_single(model, generator, e, ep_hist_train, ind)
        # Validing epoch
        validate_epoch_single(model, generator, e, ep_hist_val, ind)


def load_network():
    """
    Loads the last trained model with its setting.
    In settings are stored index of the last epoch, the best
    validation loss achieved and its index.
    :return: loaded Keras model + index of the last epoch before save
    """
    global BEST_LOSS, BEST_EPOCH_IND
    model, vars_dict = load_model(model_path)
    print(vars_dict)
    BEST_LOSS = vars_dict["loss"]
    BEST_EPOCH_IND = vars_dict["ep_ind"]
    start_ep = vars_dict["epoch"]
    return model,start_ep


def run_load_model():
    """
    Loads model from saved location and runs it.
    :return:
    """
    global BEST_LOSS,BEST_EPOCH_IND
    ep_hist_train = {}
    ep_hist_val = {}
    model,vars_dict = load_model(model_path)
    print(vars_dict)
    BEST_LOSS = vars_dict["loss"]
    BEST_EPOCH_IND = vars_dict["ep_ind"]
    start_ep = vars_dict["epoch"] + 1
    opt = optimizers.Adam(lr=LEARNING_RATE)
    # model.compile(optimizer=rms, loss=["categorical_crossentropy", "categorical_crossentropy","categorical_crossentropy", "categorical_crossentropy","categorical_crossentropy"], metrics=['accuracy'])
    # model.compile(optimizer=opt,loss= "categorical_crossentropy", metrics=['accuracy'])
    model.compile(optimizer=opt,loss=masked_loss_function, metrics=['accuracy'])
    generator = DataGeneratorCelebASparse(resolution, bulk_size)

    print("Starting loaded model at epoch[",str(start_ep),"]"," with best loss: ", str(BEST_LOSS))
    for e in range(start_ep,n_epochs):
        print("epoch %d" % e)
        # Training
        train_epoch(model, generator, e, ep_hist_train)
        # Validating
        validate_epoch(model, generator, e, ep_hist_val)


def run_load_model_single(ind):
    """
    Loads single output model from saved location and runs it.
    :return:
    """
    global BEST_LOSS,BEST_EPOCH_IND
    ep_hist_train = {}
    ep_hist_val = {}
    model,vars_dict = load_model(model_path)
    print(vars_dict)
    BEST_LOSS = vars_dict["loss"]
    BEST_EPOCH_IND = vars_dict["ep_ind"]
    start_ep = vars_dict["epoch"] + 1
    opt = optimizers.Adam(lr=LEARNING_RATE)
    # model.compile(optimizer=rms, loss=["categorical_crossentropy", "categorical_crossentropy","categorical_crossentropy", "categorical_crossentropy","categorical_crossentropy"], metrics=['accuracy'])
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
    generator = DataGeneratorCelebA(resolution, bulk_size)

    print("Starting loaded model at epoch[",str(start_ep),"]"," with best loss: ", str(BEST_LOSS))
    for e in range(start_ep,n_epochs):
        print("epoch %d" % e)
        train_epoch_single(model, generator, e, ep_hist_train, ind)
        # Validing epoch
        validate_epoch_single(model, generator, e, ep_hist_val, ind)


def run_load_model_virtual():
    """
    Loads model from saved location and runs it.
    :return:
    """
    global BEST_LOSS, BEST_EPOCH_IND
    ep_hist_train = {}
    ep_hist_val = {}
    model,vars_dict = load_model(model_path)
    print(vars_dict)
    BEST_LOSS = vars_dict["loss"]
    BEST_EPOCH_IND = vars_dict["ep_ind"]
    start_ep = vars_dict["epoch"] + 1
    opt = optimizers.Adam(lr=LEARNING_RATE)
    model.compile(optimizer=opt,loss= "categorical_crossentropy", metrics=['accuracy'])
    generator = DataGeneratorCelebAVirtual(resolution, bulk_size)

    print("Starting loaded model at epoch[",str(start_ep),"]"," with best loss: ", str(BEST_LOSS))
    for e in range(start_ep,n_epochs):
        print("epoch %d" % e)
        # Training
        train_epoch(model, generator, e, ep_hist_train)
        # Validating
        validate_epoch(model, generator, e, ep_hist_val)


def run_load_model_video_vgg():
    """
    Loads model from saved location and runs it.
    :return:
    """
    global BEST_LOSS, BEST_EPOCH_IND
    ep_hist_train = {}
    ep_hist_val = {}
    model,vars_dict = load_model_specific(model_path+"best_model_video")
    print(vars_dict)
    opt = optimizers.Adam(lr=LEARNING_RATE)

    model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=in_shape)


    output_layers = []
    x = model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dense(1024, activation="relu")(x)
    for cnt, _name in zip([2,2,2,2,5], ["atract","glas","gender","smile","hair"]):
        output_layers.append(Dense(cnt, activation='softmax', name=_name)(x))

    # creating the final model
    model_final = Model(input=model.input, output=output_layers)

    for layer in model_final.layers[:-10]:
        print(layer.name)
        layer.trainable = False

    model_final.compile(optimizer=opt,loss= "categorical_crossentropy", metrics=['accuracy'])
    generator = VideoGenerator(resolution, bulk_size)

    for e in range(10):
        print("epoch %d" % e)
        # Training
        train_epoch(model_final, generator, e, ep_hist_train)


def run_load_model_video():
    """
    Loads model from saved location and runs it.
    :return:
    """
    global BEST_LOSS, BEST_EPOCH_IND
    generator = VideoGenerator(resolution, bulk_size)
    ep_hist_train = {}
    ep_hist_val = {}
    model,vars_dict = load_model_specific(model_path+"best_model_video")
    print(vars_dict)
    opt = optimizers.Adam(lr=LEARNING_RATE)

    for layer in model.layers[:-9]:
        print(layer.name)
        layer.trainable = False

    model.compile(optimizer=opt,loss= "sparse_categorical_crossentropy", loss_weights=[0.01, 1, 1, 0.41, 0.61],metrics=['accuracy'])


    for e in range(100):
        print("epoch %d" % e)
        # Training
        train_epoch(model, generator, e, ep_hist_train)



def run_network(model, optim, generator, start_ep):
    model.compile(optimizer=optim, loss="categorical_crossentropy", metrics=['accuracy'])
    ep_hist_train = {}
    ep_hist_val = {}
    print("Starting loaded model at epoch[", str(start_ep), "]", " with best loss: ", str(BEST_LOSS))
    for e in range(start_ep, n_epochs):
        print("epoch %d" % e)
        # Training
        train_epoch(model, generator, e, ep_hist_train)
        # Validating
        validate_epoch(model, generator, e, ep_hist_val)


def run_model_virtual():
    """
    Prepares fresh new model and network and runs it
    with virtual image generator.
    :return:
    """
    model = define_network(in_shape=in_shape)
    opt = optimizers.Adam(lr=LEARNING_RATE)
    model.compile(optimizer=opt,loss= "categorical_crossentropy",loss_weights=[1, 1, 1, 1, 1], metrics=['accuracy'])
    ep_hist_train = {}
    ep_hist_val = {}
    generator = DataGeneratorCelebAVirtual(resolution, bulk_size)
    for e in range(n_epochs):
        print("epoch %d" % e)
        train_epoch(model, generator, e, ep_hist_train)
        # Validating epoch
        validate_epoch(model, generator, e, ep_hist_val)


if __name__ == "__main__":
    # issue with memory, in default tensorflow allocates nearly all possible memory
    # this can result in OOM error later
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    run_model()
    # run_model_hidden()
    # run_model_virtual()
    # RunLoadedModelWithGenerators()
    # path="histories/0epoch_train_hist.npy"
    # print(load_history(path))
    # run_load_model()
    # run_load_model_virtual()
    # run_model_with_single_out(2)
    # run_load_model_single(0)
    # run_model_wiki()
    # run_model_merged()
    # run_model_imdb()
    # run_model_adience()
    # run_load_model_video()