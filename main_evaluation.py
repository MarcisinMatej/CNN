import numpy as np
import tensorflow as tf
from keras import optimizers

from CNN import load_model, save_dictionary
from data_proc.DataGeneratorOnLineSparse import DataGeneratorOnLineSparse
from data_proc.DataLoader import get_cat_attributes_names
from main_plots import plot_history, prepare_eval_history, plot_matrix
from main_training import batch_size, model_path, bulk_size, resolution, mask_value


def eval_model(model, generator):
    predictions = []
    labels = []
    print("Wut?")
    for X_data, Y_data in generator:  # these are chunks of ~bulk pictures
        predictions = model.predict(X_data, batch_size=batch_size)
        labels = Y_data
        break

    for X_data, Y_data in generator:  # these are chunks of ~bulk pictures
        res = model.predict(X_data, batch_size=batch_size)
        print("pls")
        #todo make dynamic
        for i in range(5):
            predictions[i] = np.concatenate((predictions[i], res[i]), axis=0)
            labels[i] = np.concatenate((labels[i], Y_data[i]), axis=0)
    return predictions, labels


def eval_model_single_out(model, generator,ind):
    predictions = []
    labels = []
    print("Wut?")
    for X_data, Y_data in generator:  # these are chunks of ~bulk pictures
        predictions = model.predict(X_data, batch_size=batch_size)
        labels = Y_data[ind]
        break

    for X_data, Y_data in generator:  # these are chunks of ~bulk pictures
        res = model.predict(X_data, batch_size=batch_size)
        predictions = np.concatenate((predictions, res), axis=0)
        labels = np.concatenate((labels, Y_data[ind]), axis=0)
    return predictions, labels


def generate_dif_mat(predictions, labels, plot_flg=False,sub_set = ""):
    matrices = []
    att_cnt = len(predictions)
    #todo add iff predictions empty
    for i in range(att_cnt):
        l = len(predictions[i][0])
        s = (l, l)
        print("Shape:",s)
        matrices.append(np.zeros(shape=s))

    for att_pred, att_lab,i in zip(predictions, labels, range(att_cnt)):
        for pred, lab in zip(att_pred,att_lab):
            if lab != mask_value:
                # categorical
                p = np.argmax(pred)
                # l = np.argmax(lab)
                # matrices[i][p][l] += 1

                # sparse
                matrices[i][p][lab] += 1

    if plot_flg:
        for i in range(att_cnt):
            plot_matrix(matrices[i],str(i)+"_"+sub_set+"_",get_cat_attributes_names())
    return matrices


def generate_dif_mat_single(predictions, labels, plot_flg=False,sub_set = ""):
    """
    Generates diffusion matrix for single output
    :param predictions:
    :param labels:
    :param plot_flg:
    :param sub_set:
    :return:
    """
    matrices = []

    # single case
    for i in range(1):
        s = (np.shape(predictions)[-1], np.shape(predictions)[-1])
        matrices.append(np.zeros(shape=s))
    for pred, lab in zip(predictions, labels):
        p = np.argmax(pred)
        matrices[0][p][lab] += 1

    if plot_flg:
        for i in range(1):
            plot_matrix(matrices[i],str(i)+"_"+sub_set+"_",get_cat_attributes_names())
    return matrices


def run_difusion_matrix(_model, _generator, name):
    print("Eval")
    preds, labs = eval_model(_model, _generator.generate_validation())
    return generate_dif_mat(preds,labs,False,name)


def run_difusion_matrix_single(model, generator, ind, name):
    preds, labs = eval_model_single_out(model, generator, ind)
    return generate_dif_mat_single(preds, labs,False,name)


def test_model(_model, _generator):
    hist_tst = []
    tst_gen = _generator.generate_testing()
    for X_train, Y_train in tst_gen:  # these are chunks of ~bulk pictures
        hist_tst.append(_model.evaluate(x=X_train, y=Y_train, batch_size=batch_size))
    plot_history(prepare_eval_history(hist_tst), {}, 'Eval_testing', plot_flag=False,ser_flg=True,agg=False)


def evaluate_all(_model, _generator):
    matrices_dict = {'val': run_difusion_matrix(_model, _generator, "val"),
                     'train': run_difusion_matrix(_model, _generator, "train"),
                     'test': run_difusion_matrix(_model, _generator, "test")}
    save_dictionary(path_loc="diff_dict", dict=matrices_dict)
    test_model(_model, _generator)


def evaluate_single(_model, _generator, ind):
    matrices_dict = {'val': run_difusion_matrix_single(_model, _generator.generate_validation(), ind, "val"),
                     'train': run_difusion_matrix_single(_model, _generator.generate_training(), ind, "train"),
                     'test': run_difusion_matrix_single(_model, _generator.generate_testing(), ind, "tst")}
    save_dictionary(path_loc="diff_dict", dict=matrices_dict)


if __name__ == "__main__":
    # issue with memory, in default tensorflow allocates nearly all possible memory
    # this can result in OOM error later
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    model, vars_dict = load_model(model_path+"best_")
    opt = optimizers.Adam(lr=0.0000015)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
    generator = DataGeneratorOnLineSparse(resolution, bulk_size)

    BEST_LOSS = vars_dict["loss"]
    BEST_EPOCH_IND = vars_dict["ep_ind"]
    print("Evaluating model from epoch[", str(BEST_EPOCH_IND), "]", " with best loss: ", str(BEST_LOSS))


    # evaluate_all(model,generator)
    evaluate_single(model,generator,0)
