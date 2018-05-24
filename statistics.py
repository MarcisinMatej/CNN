"""
Various test like speed test, data statistics etc...
"""

import datetime
from collections import Counter

from keras import optimizers

from CNN import *
from data_proc import DataGeneratorCelebA
from data_proc.DataGenerator import DataGenerator
from data_proc.DataGeneratorIMDB import load_config_imdb
from data_proc.DataGeneratorWiki import load_config_wiki, CONF_FILE
from data_proc.ConfigLoaderCelebA import load_label_txts, load_folder_txts, get_cat_attributes_names
import matplotlib.pyplot as plt

from data_proc.ImageHandler import crop_resize

bulk_size = 1024
model_path = 'model/'
n_epochs = 250
batch_size = 32
in_shape = (64, 64, 3)

# colors = ['#b5ffb9', "#f9bc86", "#a3acff", "red", "blue", "red"]

colors = ["#ef7161", "#2f4858", "#a06cd5", "#32965d", "red", "blue"]

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

    model = define_network_with_BN(in_shape=in_shape)
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

def count_gender_attractivenes(data):
    att_cnt = len(data[data[:, 0] == '1'])
    att_men_cnt = len( data[(data[:, 0] == '1') & (data[:, 2] == '1') ] )
    att_fem_cnt = len(data[(data[:, 0] == '1') & (data[:, 2] == '2')])
    un_cnt = len(data) - att_cnt
    un_men_cnt = len(data[(data[:, 0] == '2') & (data[:, 2] == '1')])
    un_fem_cnt = len(data[(data[:, 0] == '2') & (data[:, 2] == '2')])
    print("Attractive people: ",att_cnt)
    print("Attractive male: ", att_men_cnt/att_cnt*100 )
    print("Attractive female: ", att_fem_cnt / att_cnt * 100)
    print("Untractive people: ", un_cnt)
    print("Unttractive male: ", un_men_cnt / un_cnt * 100)
    print("Unttractive female: ", un_fem_cnt / un_cnt * 100)


def plot_wiki_splits(data, cnt):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 22})

    r = [0, 1, 2]

    bottoms = [0, 0, 0]
    barWidth = 0.5
    for i in range(cnt):
        # plot
        # Create green Bars
        plt.bar(r, data[i], bottom=bottoms, color=colors[i], edgecolor='white', width=barWidth)
        # Create orange Bars
        bottoms = [i + j for i, j in zip(bottoms,data[i])]

    # Custom x axis
    names = ("Train", "Validation", "Test")
    plt.xticks(r, names)
    plt.xlabel("group")
    plt.yticks(np.arange(0, 111, 10))
    # Show graphic
    plt.show()


def wiki_age_plot():
    plt.rcParams.update({'font.size': 20})
    age_arr = []
    with open("data_proc/config_files/wiki_conf.txt") as f:
        lines = f.readlines()
        for line in lines:
            age = int(line.split(",")[2])
            if age < 0 or age > 100:
                continue
            age_arr.append(age)

    n_bins = 50
    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)

    # We can set the number of bins with the `bins` kwarg
    axs.hist(age_arr, bins=n_bins)

    plt.show()


def imdb_age_plot():
    age_arr = []

    with open("data_proc/config_files/imdb.txt", encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            arr = line.split("\t")
            age_arr.append(int(arr[6]))

    n_bins = 60
    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)

    # We can set the number of bins with the `bins` kwarg
    axs.hist(age_arr, bins=n_bins,color=colors[4])
    plt.show()

def imdb_cat_age_plot():
    age_arr = []

    with open("data_proc/config_files/imdb.txt", encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            arr = line.split("\t")
            # age_arr.append(int(arr[6]))
            if int(arr[6]) < 25:
                age_arr.append(1)
            elif int(arr[6]) < 31:
                age_arr.append(2)
            elif int(arr[6]) < 36:
                age_arr.append(3)
            elif int(arr[6]) < 42:
                age_arr.append(4)
            elif int(arr[6]) < 50:
                age_arr.append(5)
            else:
                age_arr.append(6)

    n_bins = 6
    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
    # We can set the number of bins with the `bins` kwarg
    axs.hist(age_arr, bins=n_bins, color="blue")
    plt.show()

def reformat_data_wiki(tr_f, data_gen, data_age):
    for dict_val in tr_f:
        if len(dict_val.keys()) == 2:
            data_gen[0].append(float(dict_val[0]))
            data_gen[1].append(float(dict_val[1]))
        else:
            for i in range(5):
                data_age[i].append(float(dict_val[i]))

def reformat_data_imdb(tr_f, data_gen, data_age):
    for dict_val in tr_f:
        if len(dict_val.keys()) == 2:
            data_gen[0].append(float(dict_val[0]))
            data_gen[1].append(float(dict_val[1]))
        else:
            for i in range(6):
                data_age[i].append(float(dict_val[i]))


def RunDataStatsWiki():
    tr, val, tst, attr_map = load_config_wiki(CONF_FILE)

    train = []
    valid = []
    test = []
    for key in tr:
        train.append(attr_map[key])
    for key in val:
        valid.append(attr_map[key])
    for key in tst:
        test.append(attr_map[key])

    print("TRAIN")
    freq_dict = count_freq(np.asarray(train), 2)
    data_gen = [[],[]]
    data_age = [[],[],[],[],[]]
    reformat_data_wiki(freq_dict, data_gen, data_age)
    print("VALIDATION")
    freq_dict = count_freq(np.asarray(valid), 2)
    reformat_data_wiki(freq_dict, data_gen, data_age)
    print("TEST")
    freq_dict = count_freq(np.asarray(test), 2)
    reformat_data_wiki(freq_dict, data_gen, data_age)

    plot_wiki_splits(data_age, 5)
    plot_wiki_splits(data_gen, 2)


def RunDataStatsImdb():
    tr, val, tst, attr_map, coords = load_config_imdb("imdb.txt")

    train = []
    valid = []
    test = []
    for key in tr:
        train.append(attr_map[key])
    for key in val:
        valid.append(attr_map[key])
    for key in tst:
        test.append(attr_map[key])

    print("TRAIN")
    freq_dict = count_freq(np.asarray(train), 2)
    data_gen = [[],[]]
    data_age = [[],[],[],[],[],[]]
    reformat_data_imdb(freq_dict, data_gen, data_age)
    print("VALIDATION")
    freq_dict = count_freq(np.asarray(valid), 2)
    reformat_data_imdb(freq_dict, data_gen, data_age)
    print("TEST")
    freq_dict = count_freq(np.asarray(test), 2)
    reformat_data_imdb(freq_dict, data_gen, data_age)

    plot_wiki_splits(data_age, 6)
    plot_wiki_splits(data_gen, 2)


def RunDataStatsCelebA():
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

    count_gender_attractivenes(train)
    count_gender_attractivenes(valid)
    count_gender_attractivenes(test)


def convert_to_percentage_mat(matrix):
    m_sum = sum(matrix)
    for row_i in range(len(matrix)):
        for col_i in range(len(matrix[row_i])):
            matrix[row_i][col_i] = matrix[row_i][col_i]/m_sum
    return matrix


def print_errs(matrix, alpha):

    for i,name in zip(range(len(matrix)),alpha):
        print(name,": ",str(int(100*(1-matrix[i][i]))))


def run_err_stats():
    """
    Computes error rate in percentage per category.
    Requires diff_dict.npy to load data.
    :return:
    """
    d_d = load_dictionary("diff_dict.npy")
    alphas = get_cat_attributes_names()
    for key in d_d.keys():
        print("----- ", key, "-------")
        for matrix, alpha in zip(d_d[key], alphas):
            print_errs(convert_to_percentage_mat(matrix), alpha)

def expand_coords(coords):
    """
    Expands coordinates by 25%
    :param coords:
    :return:
    """
    sc_coords = []
    # increase/decrease by scale, then increase borders to each direction by 25 %, convert to int
    sc_coords.append(int((coords[0]) * 0.8))
    sc_coords.append(int((coords[1]) * 0.8))
    sc_coords.append(int((coords[2]) * 1.2))
    sc_coords.append(int((coords[3]) * 1.2))
    return sc_coords

def prepare_crop_imdb():
    folder_imdb = "/datagrid/personal/marcisin/"
    with open("data_proc/config_files/imdb.txt", encoding="utf8") as f:
        lines = f.readlines()
        cnt = 0
        for line in lines:
            cnt += 1
            arr = line.split("\t")
            path = folder_imdb + arr[0]
            # coords = list(map(int, arr[2:6]))
            coords = expand_coords(list(map(int, arr[2:6])))
            saved_location = "data_proc/data/imdb/" + arr[0]
            try:
                crop_resize(path, coords, saved_location, (100, 100))
            except Exception as e:
                print(str(e))

if __name__ == "__main__":
    # issue with memory, in default tensorflow allocates nearly all possible memory
    # this can result in OOM error later
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)

    #RunModelBatchTest()
    RunDataStatsCelebA()
    # RunDataStatsWiki()
    # run_err_stats()
    # wiki_age_plot()
    # imdb_age_plot()
    # prepare_crop_imdb()
    # imdb_cat_age_plot()
    # RunDataStatsImdb()