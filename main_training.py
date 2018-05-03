import tensorflow as tf
from keras import optimizers

from CNN import save_model, load_model, define_network_with_BN, define_network
from data_proc.DataGeneratorOnLine import DataGeneratorOnLine
from data_proc.DataGeneratorOnLineSparse import DataGeneratorOnLineSparse
from data_proc.MyVirtualGenerator import MyVirtualGenerator
from main_plots import plot_history, merge_history, prepare_eval_history

bulk_size = 10240
model_path = 'models/'
n_epochs = 100
batch_size = 124
in_shape = (100, 100, 3)
resolution = (100, 100)
VIRT_GEN_STOP = 1
BEST_LOSS = 999999999
BEST_EPOCH_IND = 0
# without BN
LEARNING_RATE = 0.0000007
# with BN
# learning_rate = 0.0000007
MASK_VALUE = -1

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


from keras import backend as K
# def masked_loss_functions(target, output):
#     # mask = K.cumprod(K.cast(K.not_equal(target, mask_value), tf.int64))
#     mask = tf.reshape(K.cast(K.not_equal(target, mask_value), tf.int64), shape=(-1, 1))
#     # tf.nn.weighted_cross_entropy_with_logits()
#     # if mask == 0:
#     #     losses = K.sparse_categorical_crossentropy(output, output)
#     # else:
#     losses = K.sparse_categorical_crossentropy(target,output)
#     losses = tf.reshape(losses, shape=(-1, 1))
#     return tf.losses.compute_weighted_loss(
#         weights=mask,
#         losses=losses)

def masked_accuracy(y_true, y_pred):
    dtype = K.floatx()
    total = K.sum(K.cast(K.not_equal(y_true, MASK_VALUE), dtype))
    correct = K.sum(K.cast(K.equal(y_true, K.round(y_pred)), dtype))
    return correct / total


def masked_loss_function(target, output):
    mask = K.cast(K.not_equal(target, MASK_VALUE), K.floatx())
    return K.sparse_categorical_crossentropy(tf.multiply(target, mask), tf.multiply(output, mask))


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
        hist_val.append(model.evaluate(x=X_train, y=Y_train, batch_size=batch_size))
        agg += hist_val[-1][0]
    plot_history(prepare_eval_history(hist_val), ep_hist_val, str(epoch_id) + 'epoch_validation')
    # save model if we get better validation loss

    if agg < BEST_LOSS:
        print("!!!!!!!!!!!!!!!!!!  AGG LOSS IMPROVEMENT, now:" + str(BEST_LOSS) + ", new:" + str(agg))
        save_model(model, model_path+"best_",epoch_id,BEST_LOSS,BEST_EPOCH_IND)
        BEST_LOSS = agg
        BEST_EPOCH_IND = epoch_id
    else:
        print("AGG LOSS NOT IMPROVED, BEST:" + str(BEST_LOSS) + ", current:" + str(agg))


def train_epoch_single(model, generator, ep_ind, ep_hist_train, out_index):
    """
    Procedure to train provided model with provide data from generator
    in single epoch. After training results are plotted with plot_history(...) function.
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
        # TODO here we can select just 1 attribute for training
        histories_train.append(model.fit(X_train, Y_train[out_index], batch_size=batch_size, epochs=1))

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
        hist_val.append(model.evaluate(x=X_train, y=Y_train[out_index], batch_size=batch_size))
        agg += hist_val[-1][0]
    plot_history(prepare_eval_history(hist_val), ep_hist_val, str(epoch_id) + 'epoch_validation')
    # save model if we get better validation loss

    if agg < BEST_LOSS:
        print("!!!!!!!!!!!!!!!!!!  AGG LOSS IMPROVEMENT, now:" + str(BEST_LOSS) + ", new:" + str(agg))
        save_model(model, model_path+"best_", epoch_id, BEST_LOSS, BEST_EPOCH_IND)
        BEST_LOSS = agg
        BEST_EPOCH_IND = epoch_id


def run_model():
    """
    Prepares fresh new model and network and runs it.
    :return:
    """
    model = define_network_with_BN(in_shape=in_shape)
    opt = optimizers.Adam(lr=LEARNING_RATE)
    model.compile(optimizer=opt,loss= "sparse_categorical_crossentropy",loss_weights=[1, 1, 1, 1, 1], metrics=['accuracy'])

    ep_hist_train = {}
    ep_hist_val = {}
    generator = DataGeneratorOnLine(resolution, bulk_size)
    for e in range(n_epochs):
        print("epoch %d" % e)
        train_epoch(model, generator, e, ep_hist_train)
        # Validing epoch
        validate_epoch(model, generator, e, ep_hist_val)


def run_model_hidden():
    """
    Prepares fresh new model and network and runs it.
    :return:
    """
    model = define_network_with_BN(in_shape=in_shape)
    opt = optimizers.Adam(lr=LEARNING_RATE)
    # model.compile(optimizer=opt,loss= "sparse_categorical_crossentropy",loss_weights=[1, 1, 1, 1, 1], metrics=['accuracy'])
    model.compile(optimizer=opt, loss=masked_loss_function, loss_weights=[1, 1, 1, 1, 1],
                  metrics=['accuracy'])
    ep_hist_train = {}
    ep_hist_val = {}
    generator = DataGeneratorOnLineSparse(resolution, bulk_size)
    for e in range(n_epochs):
        print("epoch %d" % e)
        train_epoch(model, generator, e, ep_hist_train)
        # Validing epoch
        validate_epoch(model, generator, e, ep_hist_val)

def run_model_with_single_out(ind):
    """
    Prepares fresh new model and network and runs it.
    :return:
    """
    model = define_network(in_shape=in_shape, single_output_ind=ind)
    opt = optimizers.Adam(lr=LEARNING_RATE)
    model.compile(optimizer=opt,loss= "categorical_crossentropy", metrics=['accuracy'])
    ep_hist_train = {}
    ep_hist_val = {}
    generator = DataGeneratorOnLine(resolution, bulk_size)
    for e in range(n_epochs):
        print("epoch %d" % e)
        train_epoch_single(model, generator, e, ep_hist_train, ind)
        # Validing epoch
        validate_epoch_single(model, generator, e, ep_hist_val, ind)


def load_network():
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
    generator = DataGeneratorOnLineSparse(resolution, bulk_size)

    print("Starting loaded model at epoch[",str(start_ep),"]"," with best loss: ", str(BEST_LOSS))
    for e in range(start_ep,n_epochs):
        print("epoch %d" % e)
        # Training
        train_epoch(model, generator, e, ep_hist_train)
        # Validating
        validate_epoch(model, generator, e, ep_hist_val)


def run_load_model_single(ind):
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
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
    generator = DataGeneratorOnLine(resolution, bulk_size)

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
    generator = MyVirtualGenerator(resolution, bulk_size)

    print("Starting loaded model at epoch[",str(start_ep),"]"," with best loss: ", str(BEST_LOSS))
    for e in range(start_ep,n_epochs):
        print("epoch %d" % e)
        # Training
        train_epoch(model, generator, e, ep_hist_train)
        # Validating
        validate_epoch(model, generator, e, ep_hist_val)


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
    generator = MyVirtualGenerator(resolution, bulk_size)
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

    # run_model()
    # run_model_hidden()
    # run_model_virtual()
    # RunLoadedModelWithGenerators()
    # path="histories/0epoch_train_hist.npy"
    # print(load_history(path))
    run_load_model()
    # run_load_model_virtual()
    # run_model_with_single_out(0)
    # run_load_model_single(0)