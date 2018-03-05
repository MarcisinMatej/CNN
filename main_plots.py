"""
Helper functions for ploting results.
"""

import matplotlib.pyplot as plt
from keras.utils import plot_model
import numpy as np
from CNN import serialize_history, load_dictionary, history_path
import glob
import re

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
figures_path = 'figures/'

def _plot_model(model):
    """
    Produces plot of model, see keras.plot_model
    :param model:
    :return: saves plot image
    """
    plot_model(model, to_file='model.png')


def plot_loss(data, epoch_ind):
    """
    Plots loss curves from data dictionary.
    :param data: dictionary in form, where each key has loss in its name
     e.g. : 'string_loss':[list of integers]
    :param epoch_ind: index of epoch
    :return: saved plot as png file which name is starting by epoch_index
    """
    # Loss Curves
    c = 0
    plt.figure()
    ax = plt.subplot(111)
    for key in data.keys():
        if 'loss' in key:
            ax.plot(data[key], COLORS[c], linewidth=3.0, label=key)
            c = (c + 1) % len(COLORS)
            # plt.plot(history.history['loss'], 'r', linewidth=3.0)
            # plt.plot(history['loss'], 'r', linewidth=3.0)
            # plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
            # plt.plot(history['val_loss'], 'b', linewidth=3.0)

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    ax.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, -0.05))
    plt.xlabel('Chunks ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)
    plt.savefig(figures_path + epoch_ind + "_loss")
    plt.close('all')


def plot_accuracy(data, epoch_ind):
    """
        Plots accuracy curves from data dictionary.
        :param data: dictionary in form, where each key has 'acc' substring in its name
         e.g. : 'string_acc':[list of integers]
        :param epoch_ind: index of epoch
        :return: saved plot as png file which name is starting by epoch_index
        """
    # Accuracy Curves
    c = 0
    plt.figure(figsize=[8, 6])
    ax = plt.subplot(111)
    for key in data.keys():
        if 'acc' in key:
            ax.plot(data[key], COLORS[c], linewidth=3.0, label=key)
            c = (c + 1) % len(COLORS)
            # #plt.plot(history.history['acc'], 'r', linewidth=3.0)
            # plt.plot(history['acc'], 'r', linewidth=3.0)
            # #plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
            # plt.plot(history['val_acc'], 'b', linewidth=3.0)
            # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    ax.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, -0.05))
    plt.xlabel('Chunks ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)
    plt.savefig(figures_path + epoch_ind + '_acc')
    plt.close('all')


def plot_history(history, agg_history, epoch_ind,plot_flag=False,agg=False,ser_flg=True):
    """
    Produces plot of loss and accuracy per epoch for validation and training data.
    Values are taken from history.
    :param history: history returned from merge_histories(), basically dictionary of loss/accuracy.
    Contains values of metrics per bulk size
    :param agg_history: average metrics per epoch
    :param epoch_ind: index of epoch
    :return:
    """
    if plot_flag:
        # plot metrics in the last epoch
        plot_loss(history, epoch_ind)
        plot_accuracy(history, epoch_ind)
        plt.close('all')
    if agg:
        # plot avegare metrics thorugh all epochs
        agg_history = merge_epoch_history(agg_history,history)
        plot_loss(agg_history, "aggregate")
        plot_accuracy(agg_history, "aggregate")
        plt.close('all')
    if ser_flg:
        serialize_history(history,epoch_ind)

def merge_history(histories):
    """
    Helper method which merges multiple history files.
    :param histories: array of histories ordered in order of merging (needed for epoch index update)
    :return: single history dictionary
    """
    history = {}
    if len(histories) <= 0:
        return {}
    for key in histories[0].history.keys():
        history[key] = []

    for h in histories:
        for key in h.history.keys():
            history[key] += h.history[key]

    return history

def merge_epoch_history(avg_histories, epoch_hist):
    """
    Helper method which appends history of the last epoch
    to average epoch metrics dictionary.
    :param avg_histories : old history dictionary per epoch
    :param epoch_hist : new data from the last epoch
    :return: single history dictionary
    """
    if len(epoch_hist.keys()) <= 0:
        return avg_histories
    if len(avg_histories.keys()) <= 0:
        for key in epoch_hist.keys():
            avg_histories[key] = []
            #TODO remove?
            avg_histories[key].append(epoch_hist[key][0])

    for key in epoch_hist.keys():
        #average new
        avg_histories[key].append(sum(epoch_hist[key]) / len(epoch_hist[key]))
    return avg_histories


def prepare_eval_history(histories):
    """
    Helper function to transform list of results into dictionary
    where key is loss + id of output or acc (accuracy) + id of output.
    :param histories: list of histories from keras.model.eval(X,Y)
    :return: dictionary of histories
    """
    history = {}
    if len(histories) <= 0:
        return history

    att_cnt = len(histories[0])
    # first is aggragate loss, the rest is split half loss and the second half acc
    mat = np.asarray(histories)
    history['Agg_loss'] = np.asarray(mat[:, 0])

    for i in range(1, int(att_cnt/2) + 1):
        history['loss' + str(i)] = mat[:, i]
    for i in range(int(att_cnt/2) + 1, att_cnt):
        history['acc' + str(i)] = mat[:, + i]
    return history


def stringSplitByNumbers(x):
    r = re.compile('(\d+)')
    l = r.split(x)
    return [int(y) if y.isdigit() else y for y in l]


def plot_agg_epoch():
    agg_hist_train = {}
    agg_hist_val = {}
    paths = sorted(glob.glob(history_path + "*.npy"), key = stringSplitByNumbers)
    for path in paths:
        if "train" in path:
            agg_hist_train = merge_epoch_history(agg_hist_train, load_dictionary(path))
        elif "validation" in path:
            agg_hist_val = merge_epoch_history(agg_hist_val, load_dictionary(path))
            
    plot_loss(agg_hist_train,"Aggregate_train")
    plot_accuracy(agg_hist_train,"Aggregate_train")

    plot_loss(agg_hist_val, "Aggregate_validation")
    plot_accuracy(agg_hist_val, "Aggregate_validation")


def plot_all_epoch_hist():
    paths = sorted(glob.glob(history_path + "*.npy"), key=stringSplitByNumbers)
    i_t= 0
    i_v=0
    for path in paths:
        if "train" in path:
            plot_loss(load_dictionary(path), "train_" + str(i_t))
            i_t+=1
        elif "validation" in path:
            plot_loss(load_dictionary(path), "val_" + str(i_v))
            i_v+=1

if __name__ == "__main__":
    plot_agg_epoch()
    plot_all_epoch_hist()