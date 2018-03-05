"""
Varios test like speed test, data statistics etc...
"""

import datetime
from collections import Counter

from keras import optimizers

from CNN import *
from data_proc.DataGenerator import DataGenerator
from data_proc.DataLoader import load_label_txts, load_folder_txts

bulk_size = 1024
model_path = 'model/'
n_epochs = 250
batch_size = 32
in_shape = (64, 64, 3)

def bulk_time_test():
    sizes = [32,128,256,512,1024,2048,4096,8192]
    res = []
    for bulk_s in sizes:
        generator = DataGenerator((64, 64), bulk_s)
        train_gen = generator.generate_training()
        # training
        start = datetime.datetime.now()
        for X_train, Y_train in train_gen:  # these are chunks of ~bulk pictures
            break
        end = datetime.datetime.now()
        delta = end - start
        res.append(int(delta.total_seconds() * 1000))
    import matplotlib.pyplot as plt
    plt.plot(range(len(sizes)),res,'ro')
    plt.xticks(range(len(sizes)), sizes)
    plt.show()

    average = [res[i]/sizes[i] for i in range(len(res))]
    plt.plot(range(len(sizes)), average,'ro')
    plt.xticks(range(len(sizes)), sizes)
    plt.show()

def batch_time_test(model, generator):
    histories_train = []
    train_gen = generator.generate_training()
    res = []
    sizes = [16,32,48,64,128,256,512]
    # training
    for X_train, Y_train in train_gen:  # these are chunks of ~bulk pictures
        for b_size in sizes:
            start = datetime.datetime.now()
            histories_train.append(model.fit(X_train, Y_train, batch_size=b_size, epochs=1))
            end = datetime.datetime.now()
            delta = end - start
            res.append(int(delta.total_seconds() * 1000))
        break
    import matplotlib.pyplot as plt
    plt.plot(range(len(sizes)), res, 'ro')
    plt.xticks(range(len(sizes)), sizes)
    plt.show()


def RunModelBatchTest():

    model = define_network(in_shape=in_shape)
    opt = optimizers.Adam(lr=0.0000015)
    model.compile(optimizer=opt,loss= "categorical_crossentropy", metrics=['accuracy'])

    for e in range(n_epochs):
        print("epoch %d" % e)
        generator = DataGenerator((64, 64), bulk_size)
        # Training
        batch_time_test(model, generator)

def count_freq(data,cnt):
    freq_dict = []
    s = 0
    for i in range(cnt):
        x = data[:,i]
        freq_dict.append(Counter(x))
        s = sum(freq_dict[-1].values())
        for key in freq_dict[-1].keys():
            # make percentage and round on two decimal places
            freq_dict[-1][key] = format(freq_dict[-1][key]/s*100,'.2f')
        print(freq_dict[-1])

    print("Sum:" + str(s))
    return freq_dict

def RunDataStats():
    attr_vals, lbs_map = load_label_txts()
    train = []
    valid = []
    test = []
    for line in load_folder_txts():
        key = line.split()[0].split("/")[-1]
        val = line.split()[1]
        if val == "1":
            train.append(lbs_map[key])
        elif val == "2":
            test.append(lbs_map[key])
        elif val == "3":
            valid.append(lbs_map[key])

    print("TRAIN")
    count_freq(np.asarray(train),len(attr_vals))
    print("VALIDATION")
    count_freq(np.asarray(valid), len(attr_vals))
    print("TEST")
    count_freq(np.asarray(test), len(attr_vals))


if __name__ == "__main__":
    # issue with memory, in default tensorflow allocates nearly all possible memory
    # this can result in OOM error later
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)

    #RunModelBatchTest()

    RunDataStats()