import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns

# Fixing random state for reproducibility
# np.random.seed(19680801)
#
#
# N = 50
# x = np.random.rand(N)
# y = np.random.rand(N)
# colors = np.random.rand(N)
# area = (30 * np.random.rand(N))**2  # 0 to 15 point radii
# print(area)
#
# plt.scatter(x, y, s=area, c=colors, alpha=0.5)
# plt.show()

def UCF101():
    title = ['(2+1)D ShuffleNet', 'R2+1D-18', 'STC-ResNeXt', 'T3D-169', 'I3D',
             '3D ResNet-34', '3D ResNet-50',
             '3D ResNet-101',
             '3D ResNeXt-101', 'IIC']
    acc = [86.4, 85.7, 84.2, 81.3, 84.5, 87.7, 89.3, 88.9, 90.7, 74.4]
    Flops = [12.1, 31.8, 56.1, 100.3, 107.9, 54.0, 77.3, 84.7, 64.7, 42.6]
    params = [17.1, 26.1, 184.0, 42.7, 25, 46.9, 63.7, 86, 164.8, 33.4]
    colors = ['red', 'chocolate', 'darkorange', 'darkkhaki', 'darkseagreen', 'olive',
              'darkcyan', 'steelblue', 'navy', 'deeppink']
    s = (np.asarray(params))** 2 // 6

    plt.scatter(Flops, acc, s=s, c=colors, alpha=0.5, label=title)
    for (x,y, l) in zip(Flops, acc, title):
        plt.annotate(l, (x,y),textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center')
    # plt.legend()
    # plt.legend(title, scatterpoints=1, loc='lower left', ncol=3, fontsize=8)

    plt.gca()
    plt.grid()
    plt.ylabel('Accuracy', fontsize=15)
    plt.xlabel('GFLOPs', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()

def UCF101_seaborn():
    title = ['(2+1)D ShuffleNet', 'R2+1D-18', 'STC-ResNeXt', 'T3D-169', 'I3D',
             '3D ResNet-34', '3D ResNet-50',
             '3D ResNet-101',
             '3D ResNeXt-101', 'IIC']
    acc = [86.4, 85.7, 84.2, 81.3, 84.5, 87.7, 89.3, 88.9, 90.7, 74.4]
    Flops = [12.1, 31.8, 56.1, 100.3, 107.9, 54.0, 77.3, 84.7, 64.7, 42.6]
    params = [17.1, 26.1, 184.0, 42.7, 25, 46.9, 63.7, 86, 164.8, 33.4]
    colors = ['red', 'chocolate', 'darkorange', 'darkkhaki', 'darkseagreen', 'olive',
              'darkcyan', 'steelblue', 'navy', 'deeppink']
    s = (np.asarray(params))** 2 // 10
    # s = (np.asarray(params)) * 10
    s = np.asarray(s, dtype=int)

    fig, ax = plt.subplots()
    scatter = ax.scatter(Flops, acc, c=colors, s=s)

    # produce a legend with the unique colors from the scatter
    for (x,y,l,r) in zip(Flops, acc, title, s):
        print(math.sqrt(r))
        plt.annotate(l, (x,y),textcoords="offset points", # how to position the text
                 xytext=(0,math.sqrt(r)/2 + 5), # distance from text to points (x,y)
                 ha='center')
    # plt.legend()
    # plt.legend(title, scatterpoints=1, loc='lower left', ncol=3, fontsize=8)

    plt.gca()
    plt.grid()
    plt.ylabel('Accuracy', fontsize=15)
    plt.xlabel('GFLOPs', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim([74, 93])

    # produce a legend with a cross section of sizes from the scatter
    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.1, func=lambda s: np.sqrt(s))
    print(s)
    print(labels)
    labels = ['20M', '40M', '60M', '80M', '100M ', '120M', ' 140M', '  160M', '   180M']
    legend2 = ax.legend(handles, labels, title="#Params", ncol=9, frameon=False, fontsize=10, title_fontsize=12, columnspacing=1.5,
                        loc='lower right', bbox_to_anchor=(0.5, 0.05, 0.5, 0.5))
    # legend2 = ax.legend(handles, labels, title="#Params", ncol=9, fontsize=12, title_fontsize=12, borderpad=1.15, frameon=False)
    # legend2 = ax.legend(handles, labels, title="#Params", ncol=9, prop={'size':18})
    # legend2 = ax.legend(handles, labels, title="#Params", ncol=9, borderpad=1.8, columnspacing=1.3)

    plt.show()

def HMDB51_seaborn():
    title = ['(2+1)D ShuffleNet', 'R2+1D-18', 'T3D-169', 'I3D',
             '3D ResNet-34', '3D ResNet-50',
             '3D ResNet-101',
             '3D ResNeXt-101', 'IIC']
    acc = [59.9, 54.9, 61.1, 49.8, 59.1, 61.0, 61.7, 63.8, 38.3]
    Flops = [12.1, 31.8, 100.3, 107.9, 54.0, 77.3, 84.7, 64.7, 42.6]
    params = [17.1, 26.1, 42.7, 25, 46.9, 63.7, 86, 164.8, 33.4]
    colors = ['red', 'chocolate', 'darkorange', 'darkkhaki', 'darkseagreen', 'olive',
              'darkcyan', 'steelblue', 'navy']
    s = (np.asarray(params))** 2 // 10
    # s = (np.asarray(params)) * 10
    s = np.asarray(s, dtype=int)

    fig, ax = plt.subplots()
    scatter = ax.scatter(Flops, acc, c=colors, s=s)

    # produce a legend with the unique colors from the scatter
    for (x,y,l,r) in zip(Flops, acc, title, s):
        print(math.sqrt(r))
        plt.annotate(l, (x,y),textcoords="offset points", # how to position the text
                 xytext=(0,math.sqrt(r)/2 + 5), # distance from text to points (x,y)
                 ha='center')
    # plt.legend()
    # plt.legend(title, scatterpoints=1, loc='lower left', ncol=3, fontsize=8)

    plt.gca()
    plt.grid()
    plt.ylabel('Accuracy', fontsize=15)
    plt.xlabel('GFLOPs', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim([35, 70])

    # produce a legend with a cross section of sizes from the scatter
    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.1, func=lambda s: np.sqrt(s))
    print(s)
    print(labels)
    labels = ['20M', '40M', '60M', '80M', '100M', '120M', ' 140M', '  160M', '   180M']
    # legend2 = ax.legend(handles, labels, title="#Params", ncol=9)
    legend2 = ax.legend(handles, labels, title="#Params", ncol=9, frameon=False, fontsize=10, title_fontsize=12,
                        columnspacing=1.5)

    plt.show()


UCF101_seaborn()
# temp()
# HMDB51_seaborn()