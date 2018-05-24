from data_proc import DataGeneratorCelebA

CONF_FILE = "data_proc/config_files/ADIENCE.csv"


def load_config_imdb(expand=True):
    """
    Load configuration for Adience dataset.
    :param expand: expand bounding boxes
    :return: Train, Test, Validation ids for images from dataset +
    attribute map (dictionary: key = image name, value = array of labels)
    """
    train = set()
    val = set()
    test = set()
    attr_map = {}
    coord_dict = {}
    with open(CONF_FILE, encoding="utf8") as f:
        lines = f.readlines()
        first = True
        for line in lines:
            if first:
                # skipp header
                first = False
                continue
            arr = line.split(",")
            key = arr[0]
            ll = []
            ll.append(arr[1])
            ll.append(arr[2])
            ll.append(arr[5])
            ll.append(arr[6])
            if expand:
                coord_dict[key] = DataGeneratorCelebA.expand_coords(list(map(int, ll)))
            else:
                coord_dict[key] = list(map(int, ll))
            gender_i = 0
            if arr[11] == 'F':
                gender_i = 1

            if arr[9] == "0":
                age_cat = 0#2
            elif arr[9] == "4":
                age_cat = 1#6
            elif arr[9] == "8":
                age_cat = 2#12
            elif arr[9] == "15":
                age_cat = 3#20
            elif arr[9] == "25":
                age_cat = 4#30
            elif arr[9] == "38":
                age_cat = 5#43
            elif arr[9] == "48":
                age_cat = 6#60
            else:
                age_cat = 7

            attr_map[key] = [gender_i, age_cat]

            if arr[-1] == "2\n":
                val.add(key)
            elif arr[-1] == "3\n":
                test.add(key)
            else:
                train.add(key)

    print("---Training set has len: ", str(len(train)))
    print("---Testing set has len: ", str(len(test)))
    print("---Validation set has len: ", str(len(val)))
    return list(train), list(val), list(test), attr_map, coord_dict
