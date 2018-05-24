"""
Script for optimization of Learning rate.
"""

import random
from keras import optimizers
from CNN import  define_network_BN_multi
from data_proc.DataGeneratorOUI import DataGeneratorOUI
from main_training import resolution, VERBOSE, in_shape, bulk_size

LR_BEST = 6.95309166612175e-05

def run_model_adience():
    """
    Prepares fresh new model for imdb dataset
    and trains it.
    :return:
    """
    BEST = 0
    BEST_LR = 0
    BS = 62
    epochs = 100
    generator = DataGeneratorOUI(resolution, bulk_size)
    lrs = []
    results = []
    for exp in range(100):
        LR = random.uniform(9.249818188607698e-08, 1.001e-07) #3.2249965267539267e-07)
        # LR  = 3.2249965267539267e-04
        # LR = (1+random.random()) / (10**exp)
        # LR = random.uniform(5.0e-06, 1.0e-04)  # 3.2249965267539267e-07)
        # LR = 1.2249965267539267e-06
        lrs.append(LR)
        print("Testing LR = ", LR)
        opt = optimizers.Adam(lr=LR)
        # model = VGG_16([2, 8], ["Gender", "Age"], in_shape=in_shape)
        model = define_network_BN_multi([2, 8], ["Gender", "Age"], in_shape=in_shape)
        model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=['accuracy'], loss_weights=[1, 1])

        ep_hist_train = {}
        ep_hist_val = {}
        for e in range(epochs):
            histories_train = []
            train_gen = generator.generate_training()

            # training
            for X_train, Y_train in train_gen:  # these are chunks of ~bulk pictures
                # TODO here we can select just 1 attribute for training
                histories_train.append(model.fit(X_train, Y_train, batch_size=BS, epochs=1, verbose=VERBOSE))

            # plot_history(merge_history(histories_train), ep_hist_train, "A"+str(e) + 'epoch_train_LR_OPT' + str(exp))

            # Validate
            # hist_val = []
            for X_train, Y_train in generator.generate_validation():  # these are chunks of ~bulk pictures
                results.append(model.evaluate(x=X_train, y=Y_train, batch_size=BS, verbose=VERBOSE))
                    # hist_val.append(model.evaluate(x=X_train, y=Y_train, batch_size=batch_size, verbose=VERBOSE))
                # plot_history(prepare_eval_history(hist_val), ep_hist_val, str(e) + 'epoch_val_LR_OPT' + str(_))

            print("LR: ", str(LR))
            print("res: ", str(results[-1]))

    for res, lr in zip(results, lrs):
        print("LR: ", str(lr))
        print("res: ", str(res))
        print("-------------------")


if __name__ == '__main__':
    run_model_adience()