"""
Support script just for producing plots for category percentages.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_cat_bars_1():
    N = 3
    att = (51, 50, 53)
    ugly = (49, 50, 48)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.52  # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, att, width, align="center",color = "dodgerblue")
    p2 = plt.bar(ind, ugly, width, align="center", bottom=att,color = "orange")

    plt.ylabel('Percentage')
    plt.title('Percentage of attractive categories in datasets')
    plt.xticks(ind, ('Training', 'Validation', 'Testing'))
    plt.yticks(np.arange(0, 101, 10))
    plt.legend((p1[0], p2[0]), ('Attractive', 'Unattractive'))

    plt.show()

def plot_cat_bars_2():

    N = 3
    att = (6, 6, 7)
    ugly = (94, 94, 93)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.52  # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, att, width, align="center",color = "dodgerblue")
    p2 = plt.bar(ind, ugly, width, align="center", bottom=att,color = "orange")

    plt.ylabel('Percentage')
    plt.title('Percentage of people with glasses')
    plt.xticks(ind, ('Training', 'Validation', 'Testing'))
    plt.yticks(np.arange(0, 101, 10))
    plt.legend((p1[0], p2[0]), ('Withtttttttt', 'Without'))

    plt.show()

def plot_cat_bars_3():

    N = 3
    att = (42, 39, 43)
    ugly = (58, 61, 57)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.52  # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, att, width, align="center",color = "dodgerblue")
    p2 = plt.bar(ind, ugly, width, align="center", bottom=att,color = "orange")

    plt.ylabel('Percentage')
    plt.title('Percentage of genders')
    plt.xticks(ind, ('Training', 'Validation', 'Testing'))
    plt.yticks(np.arange(0, 101, 10))
    plt.legend((p1[0], p2[0]), ('Male', 'Female'))

    plt.show()

def plot_cat_bars_4():

    N = 3
    att = (48, 50, 48)
    ugly = (52, 50, 52)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.52  # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, att, width, align="center",color = "dodgerblue")
    p2 = plt.bar(ind, ugly, width, align="center", bottom=att,color = "orange")

    plt.ylabel('Percentage')
    plt.title('Percentage of smiling persons in datasets')
    plt.xticks(ind, ('Training', 'Validation', 'Testing'))
    plt.yticks(np.arange(0, 101, 10))
    plt.legend((p1[0], p2[0]), ('Yes', 'No'))

    plt.show()

def plot_cat_bars_5():

    N = 3
    cat_1 = (23, 27, 20)
    cat_2 = (14, 13, 14)
    cat_3 = (19, 17, 23)
    cat_4 = (5, 3, 5)
    cat_5 = (39, 40, 38)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.52  # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, cat_1, width, align="center",color = "navy")
    p2 = plt.bar(ind, cat_2, width, align="center", bottom=cat_1,color = "gold")
    p3 = plt.bar(ind, cat_3, bottom=[i + j for i, j in zip(cat_1, cat_2)], color='mediumseagreen', align="center",width=width)
    p4 = plt.bar(ind, cat_4, bottom=[i + j + k for i, j, k in zip(cat_1, cat_2, cat_3)], color='silver',width=width, align="center")
    p5 = plt.bar(ind, cat_5, bottom=[i + j + k + l for i, j, k, l in zip(cat_1, cat_2, cat_3, cat_4)], color='palevioletred', width=width, align="center")

    plt.ylabel('Percentage')
    plt.title('Percentage of hair colors in datasets')
    plt.xticks(ind, ('Training', 'Validation', 'Testing'))
    plt.yticks(np.arange(0, 101, 10))
    plt.legend((p1[0], p2[0],p3[0],p4[0],p5[0]), ('Black', 'Blond', 'Brown', 'Gray','Other'))

    plt.show()


def plot_all_errors():
    # libraries
    import numpy as np
    import matplotlib.pyplot as plt

    # set width of bar
    barWidth = 0.2

    # set height of bar
    errs_single = [22, 1, 3, 8, 33]
    errs_multi = [22, 1, 4, 9, 35]
    errs_hum = [30, 1, 2, 12, 48]
    errs_rand = [48, 7, 43, 48, 62]

    # Set position of bar on X axis
    r1 = np.arange(len(errs_single))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]

    # Make the plot
    plt.bar(r1, errs_single, color='#ef7161', width=barWidth, edgecolor='white', label='STM')
    plt.bar(r2, errs_multi, color='#2f4858', width=barWidth, edgecolor='white', label='MTM')
    plt.bar(r3, errs_hum, color='#a06cd5', width=barWidth, edgecolor='white', label='CV human')
    plt.bar(r4, errs_rand, color='#32965d', width=barWidth, edgecolor='white', label='Random')

    # Add xticks on the middle of the group bars
    # plt.xlabel('Attribute', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(errs_single))], ['Attractive', 'Glass', 'Gender', 'Smile', 'Hair'])

    # Create legend & Show graphic
    plt.legend(bbox_to_anchor=(0.6875, 1))
    plt.show()


if __name__ == "__main__":
    # plot_cat_bars_1()
    # plot_cat_bars_2()
    # plot_cat_bars_3()
    # plot_cat_bars_4()
    # plot_cat_bars_5()

    plot_all_errors()