from keras import optimizers

from CNN import load_model
from data_proc.DataGenerator import DataGenerator
from data_proc.DataLoader import get_cat_attributes_names
from main_plots import plot_history, prepare_eval_history, plot_matrix
from main_training import batch_size, model_path, bulk_size
import numpy as np
import tensorflow as tf


def eval_model(model, generator):
    predictions = []
    labels = []
    for X_data, Y_data in generator:  # these are chunks of ~bulk pictures
        predictions = model.predict(X_data, batch_size=batch_size)
        labels = Y_data
        break

    for X_data, Y_data in generator:  # these are chunks of ~bulk pictures
        res = model.predict(X_data, batch_size=batch_size)
        #todo make dynamic
        for i in range(5):
            predictions[i] = np.concatenate((predictions[i], res[i]), axis=0)
            labels[i] = np.concatenate((labels[i], Y_data[i]), axis=0)
    return predictions, labels


def generate_dif_mat(predictions, labels, plot_flg=False,sub_set = ""):
    matrices = []
    att_cnt = len(predictions)
    #todo add iff predictions empty
    for i in range(att_cnt):
        s = (len(predictions[i][0]), len(predictions[i][0]))
        matrices.append(np.zeros(shape=s))

    for att_pred, att_lab,i in zip(predictions, labels, range(att_cnt)):
        for pred, lab in zip(att_pred,att_lab):
            p = np.argmax(pred)
            l = np.argmax(lab)
            matrices[i][p][l] += 1
    if plot_flg:
        for i in range(att_cnt):
            plot_matrix(matrices[i],str(i)+"_"+sub_set+"_",get_cat_attributes_names())


def run_difusion_matrix_validation(model, generator):
    preds, labs = eval_model(model, generator.generate_validation())
    generate_dif_mat(preds,labs,True,"val")


def run_difusion_matrix_train(model, generator):
    preds, labs = eval_model(model, generator.generate_training())
    generate_dif_mat(preds, labs,True,"train")


def run_difusion_matrix_test(model, generator):
    preds, labs = eval_model(model, generator.generate_testing())
    generate_dif_mat(preds, labs,True,"tst")


def test_model(model, generator):
    hist_tst = []
    tst_gen = generator.generate_testing()
    for X_train, Y_train in tst_gen:  # these are chunks of ~bulk pictures
        hist_tst.append(model.evaluate(x=X_train, y=Y_train, batch_size=batch_size))
    plot_history(prepare_eval_history(hist_tst), {}, 'testing', agg=False)


if __name__ == "__main__":
    # issue with memory, in default tensorflow allocates nearly all possible memory
    # this can result in OOM error later
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    model, dict_vars = load_model(model_path+"best_")
    opt = optimizers.Adam(lr=0.0000015)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
    generator = DataGenerator((64, 64), bulk_size)

    run_difusion_matrix_validation(model, generator)
    run_difusion_matrix_train(model, generator)
    run_difusion_matrix_test(model, generator)
    test_model(model, generator)