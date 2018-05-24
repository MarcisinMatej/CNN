
CONF_FILE = "wiki_cat.txt"

def load_config_wiki_age():
    train = []
    val = []
    test = []
    tmp = {}
    with open("data_proc/config_files/"+CONF_FILE) as f:
        lines = f.readlines()
        for line in lines:
            age = int(line.split(",")[2])
            if age < 0 or age > 100:
                continue
            tmp[line.split(",")[0]] = [int(line.split(",")[1]), age]
            if line.split(",")[-1] == "0\n":
                train.append(line.split(",")[0])
            if line.split(",")[-1] == "1\n":
                val.append(line.split(",")[0])
            if line.split(",")[-1] == "2\n":
                test.append(line.split(",")[0])
    return train, val, test, tmp


def load_config_wiki(conf_file):
    train = []
    val = []
    test = []
    tmp = {}
    with open("data_proc/config_files/"+conf_file) as f:
        lines = f.readlines()
        for line in lines:
            age = int(line.split(",")[2])
            tmp[line.split(",")[0]] = [int(line.split(",")[1]), age]
            if line.split(",")[-1] == "0\n":
                train.append(line.split(",")[0])
            if line.split(",")[-1] == "1\n":
                val.append(line.split(",")[0])
            if line.split(",")[-1] == "2\n":
                test.append(line.split(",")[0])
    return train, val, test, tmp