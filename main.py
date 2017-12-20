from CNN import *
from keras import optimizers
from data_proc.data_loader import DataGenerator

def RunModel(train_data,train_labels_one_hot,test_data,test_labels_one_hot):
    model = createModel()
    batch_size = 200
    epochs = 100
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, train_labels_one_hot, batch_size=batch_size, epochs=epochs, verbose=1,
                         validation_data=(test_data, test_labels_one_hot))

    # Plot training progress
    PlotHistory(history)

    # Score trained model.
    scores = model.evaluate(test_data, test_labels_one_hot)
    # scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

def RunModelWithGenerators():
    bulk_size = 1000
    train_data_dir = 'data/train'
    validation_data_dir = 'data/validation'
    nb_train_samples = 2000
    nb_validation_samples = 800
    runs = 50
    batch_size = 32
    model = createModel()
    # PlotModel(model)
    # model.compile(optimizer='rmsprop', loss=['categorical_crossentropy', 'categorical_crossentropy'],
    #               loss_weights=[1, 1], metrics=['accuracy'])
#"categorical_crossentropy", "categorical_crossentropy","categorical_crossentropy", "categorical_crossentropy","categorical_crossentropy"
    opt = optimizers.Adam(lr=0.0001)
    # model.compile(optimizer=rms, loss=["categorical_crossentropy", "categorical_crossentropy","categorical_crossentropy", "categorical_crossentropy","categorical_crossentropy"], metrics=['accuracy'])
    model.compile(optimizer=opt,loss= "categorical_crossentropy", metrics=['accuracy'])

    tmp = DataGenerator((32,32),bulk_size)
    gen = tmp.TrainingGenerator()

    histories = []
    for e in range(runs):
        print("epoch %d" % e)
        for X_train, Y_train in gen:  # these are chunks of ~100 pictures
            print(X_train.shape)
            #TODO here we can select just 1 attribute for training
            histories.append(model.fit(X_train, Y_train, batch_size=batch_size, epochs=1))

if __name__ == "__main__":
    RunModelWithGenerators()