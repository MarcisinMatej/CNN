

encode = {"Attractive":"1", "Unattractive":"2",
          "Smiling":"1", "NotSmiling":"2",
          "No glasses":"2", "Glasses":"1",
          "Female":"2", "Male":"1",
          "Black hair":"1" , "Blond hair":"2", "Brown hair":"3", "Gray hair":"4", "Other":"5"
          }

encode_cat = {"1":0, "2":3, "3":1, "4":2, "5":4}


error_cnt = [0,0,0,0,0]

if __name__ == "__main__":
    labels = {}
    with open("100_labs.txt") as f:
        for line in f.readlines():
            labels[line.split()[0]] = line.split()[1:]
    f.close()

    resp_cnt = 0
    with open("output.csv") as f:
        for line in f.readlines():
            if "submit," in line:
                resp_cnt += 1
            elif line.startswith('0'):
                key = line.split('_')[0]
                cat_ind = encode_cat[line.split(",")[0][-1]]
                val = encode[line.split(',')[1].strip()]

                if labels[key][cat_ind] != val:
                    error_cnt[cat_ind] += 1

    print(resp_cnt)
    for err in error_cnt:
        print(err/resp_cnt)