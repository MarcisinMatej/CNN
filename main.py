from CNN import *
from keras import optimizers
from data_proc.DataGenerator import DataGenerator


def RunModel(train_data,train_labels_one_hot,test_data,test_labels_one_hot):
    model = create_model()
    batch_size = 200
    epochs = 100
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, train_labels_one_hot, batch_size=batch_size, epochs=epochs, verbose=1,
                         validation_data=(test_data, test_labels_one_hot))

    # Plot training progress
    plot_history(history)

    # Score trained model.
    scores = model.evaluate(test_data, test_labels_one_hot)
    # scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

def RunModelWithGenerators():
    bulk_size = 1024
    train_data_dir = 'data/train'
    validation_data_dir = 'data/validation'
    nb_train_samples = 2000
    nb_validation_samples = 800
    n_epochs = 250
    batch_size = 32
    in_shape = (64,64,3)
    model = create_model(in_shape=in_shape)
    # model.compile(optimizer='rmsprop', loss=['categorical_crossentropy', 'categorical_crossentropy'],
    #               loss_weights=[1, 1], metrics=['accuracy'])
#"categorical_crossentropy", "categorical_crossentropy","categorical_crossentropy", "categorical_crossentropy","categorical_crossentropy"
    opt = optimizers.Adam(lr=0.0000015)
    # model.compile(optimizer=rms, loss=["categorical_crossentropy", "categorical_crossentropy","categorical_crossentropy", "categorical_crossentropy","categorical_crossentropy"], metrics=['accuracy'])
    model.compile(optimizer=opt,loss= "categorical_crossentropy", metrics=['accuracy'])

    histories_train = []
    histories_test = []

    for e in range(n_epochs):
        print("epoch %d" % e)
        tmp = DataGenerator((64, 64), bulk_size)
        train_gen = tmp.generate_training()
        # training
        for X_train, Y_train in train_gen:  # these are chunks of ~bulk pictures
            #TODO here we can select just 1 attribute for training
            histories_train.append(model.fit(X_train, Y_train, batch_size=batch_size, epochs=1))
            #TODO debug
        plot_history(merge_history(histories_train), 'epoch' + str(e))
        # Testing
        test_gen = tmp.generate_testing()
        for X_train, Y_train in test_gen:  # these are chunks of ~bulk pictures
            histories_test.append(model.predict(X_train, Y_train, batch_size=batch_size, epochs=1))

        plot_history(merge_history(histories_test), 'epoch' + str(e))


if __name__ == "__main__":
    RunModelWithGenerators()