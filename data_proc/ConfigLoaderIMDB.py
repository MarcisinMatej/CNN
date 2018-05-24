from data_proc.DataGeneratorCelebA import DataGeneratorCelebA

CONF_FILE = "data_proc/config_files/imdb.txt"

def load_config_imdb(expand=True):
    train = set()
    val = set()
    test = set()
    attr_map = {}
    coord_dict = {}
    with open(CONF_FILE, encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            arr = line.split("\t")
            key = arr[0]
            # if invalid_img(IMAGES_FOLDER_IMDB + key):
            #     continue
            if expand:
                coord_dict[key] = DataGeneratorCelebA.expand_coords(list(map(int, arr[2:6])))
            else:
                coord_dict[key] = list(map(int, arr[2:6]))
            gender_i = 0
            if arr[7] == 'F':
                gender_i = 1

            if int(arr[6]) < 24:
                age_cat = 0
            elif int(arr[6]) < 30:
                age_cat = 1
            elif int(arr[6]) < 36:
                age_cat = 2
            elif int(arr[6]) < 42:
                age_cat = 3
            elif int(arr[6]) < 50:
                age_cat = 4
            else:
                age_cat = 5

            attr_map[key] = [gender_i, age_cat]
            # attr_map[key] = [gender_i, int(arr[6])]

            if arr[-1] == "1\n":
                train.add(key)
            if arr[-1] == "2\n":
                val.add(key)
            if arr[-1] == "3\n":
                test.add(key)

    print("---Training set has len: ", str(len(train)))
    print("---Testing set has len: ", str(len(test)))
    print("---Validation set has len: ", str(len(val)))
    return list(train), list(val), list(test), attr_map, coord_dict
