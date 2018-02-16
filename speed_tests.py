import datetime
import random

from CNN import *
from keras import optimizers
from data_proc.DataGenerator import DataGenerator
import tensorflow as tf

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



if __name__ == "__main__":
    # issue with memory, in default tensorflow allocates nearly all possible memory
    # this can result in OOM error later
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)

    #RunModelBatchTest()

    # bulk_time_test()
    k = [4.2, 3.3, 2.9, 2.6, 2.4, 2.1, 2.0, 1.9, 1.7, 1.4, 1.2]
    # k = [4.0, 3.0, 2.6, 2.2, 2.1, 1.9, 1.7, 1.6, 1.4, 1.2,1.0]
    tmp = []
    for i in range(len(k)):
        if i == 0:
            tmp.append(k[i] + (random.random() - 0.5) * 0.1)
        else:
            if (i > 5):
                for j in range(5):
                    tmp.append(k[i] + (random.random() - 0.5) * 0.05)
            else:
                for j in range(10):
                    tmp.append(k[i] + (random.random() - 0.5) * 0.2)
    plt.plot(tmp, linewidth=2)
    plt.show()