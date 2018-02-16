from keras import optimizers

from CNN import load_model, plot_history, prepare_eval_history
from data_proc.DataGenerator import DataGenerator
from main_training import batch_size, model_path, bulk_size, figures_path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def difusion_matrix(model,generator):
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
            predictions[i] = np.concatenate((predictions[i], res[i]),axis=0)
            labels[i] = np.concatenate((labels[i], Y_data[i]),axis=0)
    return predictions,labels


def generate_dif_mat(predictions,labels):
    matrices = []
    att_cnt = len(predictions)
    #todo add iff predictions empty
    for i in range(att_cnt):
        s = (len(predictions[i][0]), len(predictions[i][0]))
        matrices.append(np.zeros(shape=s))

    for att_pred,att_lab,i in zip(predictions,labels,range(att_cnt)):
        for pred, lab in zip(att_pred,att_lab):
            p = np.argmax(pred)
            l = np.argmax(lab)
            matrices[i][p][l] += 1

    for i in range(att_cnt):
        show_matrix(matrices[i], i)
    plt.close("all")


def show_matrix(matrix,att_ind):
    fig, ax = plt.subplots()
    ax.matshow(matrix, cmap=plt.cm.Blues)
    plt.xlabel("Predictions")
    plt.ylabel("True labels")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            c = int(matrix[j, i]/sum(matrix[j,:])*100)
            ax.text(i, j, str(c)+"%", va='center', ha='center')

    # plt.show()
    plt.savefig(figures_path + "confusions/att_" + str(att_ind))


def run_difusion_matrix_validation(model,generator):
    preds,labs = difusion_matrix(model, generator.generate_validation())
    generate_dif_mat(preds,labs)


def run_difusion_matrix_train(model,generator):
    preds,labs = difusion_matrix(model, generator.generate_training())
    generate_dif_mat(preds,labs)


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

    model = load_model(model_path)
    opt = optimizers.Adam(lr=0.0000015)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
    generator = DataGenerator((64, 64), bulk_size)

    run_difusion_matrix_validation(model,generator)
    run_difusion_matrix_train(model,generator)
    test_model(model,generator)