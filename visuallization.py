import numpy as np
import matplotlib.pyplot as plt

def UCF101():
    # UCF101
    shuffle_net = [86.4,  17.1, 12.1]
    r2p1d_18 = [85.7, 26.1, 31.8]
    # stc_resNext_101 = [84.2, 184.0, 56.1]
    t3d_169 = [81.3, 42.7, 100.3]
    i3d = [84.5, 25.0, 107.9]
    resnet_3d_34 = [87.7, 46.9, 54]
    resnet_3d_50 = [89.3, 63.7, 77.3]
    resnet_3d_101 = [88.9, 86., 84.7]
    # resnext_3d_101 = [90.7, 164.8, 64.7]
    IIC = [74.4, 33.4, 42.6]

    title = ['(2+1)D ShuffleNet', 'R2+1D-18', 'STC-ResNeXt', 'T3D-169', 'I3D', '3D ResNet-34', '3D ResNet-50', '3D ResNet-101',
             '3D ResNeXt-101', 'IIC']
    value = [[86.4,  17.1, 12.1], [85.7, 26.1, 31.8], [84.2, 184.0, 56.1], [81.3, 42.7, 100.3], [84.5, 25.0, 107.9],
             [87.7, 46.9, 54], [89.3, 63.7, 77.3], [88.9, 86., 84.7], [90.7, 164.8, 64.7], [74.4, 33.4, 42.6]
             ]

    # rng = np.random.RandomState(0)
    plt.subplot(121)
    i = 0
    for marker in ['o', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
        plt.plot(value[i][1], value[i][0], marker, label=title[i], markersize=18)
        i +=1
    plt.legend(numpoints=1, loc='lower right', fontsize=14)
    plt.gca()
    plt.grid()
    plt.ylabel('Accuracy', fontsize=15)
    plt.xlabel('#Params', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.subplot(122)
    i = 0
    for marker in ['o', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
        plt.plot(value[i][2], value[i][0], marker, label=title[i], markersize=18)
        i +=1
    plt.legend(numpoints=1, loc='upper left', fontsize=14)
    plt.gca()
    plt.grid()
    plt.ylabel('Accuracy', fontsize=15)
    plt.xlabel('#GFLOPs', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.show()

def hmdb51():
    # HMDB51
    shuffle_net = [59.9, 17.1, 12.1]
    r2p1d_18 = [54.9, 26.1, 31.8]
    # stc_resNext_101 = [84.2, 184.0, 56.1]
    t3d_169 = [61.1, 42.7, 100.3]
    i3d = [49.8, 25.0, 107.9]
    resnet_3d_34 = [59.1, 46.9, 54]
    resnet_3d_50 = [61.0, 63.7, 77.3]
    resnet_3d_101 = [61.7, 86., 84.7]
    # resnext_3d_101 = [63.8, 164.8, 64.7]
    IIC = [38.3, 33.4, 42.6]

    title = ['(2+1)D ShuffleNet', 'R2+1D-18', 'T3D-169', 'I3D', '3D ResNet-34', '3D ResNet-50',
             '3D ResNet-101',
             '3D ResNeXt-101', 'IIC']
    value = [[59.9, 17.1, 12.1], [54.9, 26.1, 31.8], [61.1, 42.7, 100.3], [49.8, 25.0, 107.9],
             [59.1, 46.9, 54], [61.0, 63.7, 77.3], [61.7, 86., 84.7], [63.8, 164.8, 64.7], [38.3, 33.4, 42.6]
             ]

    # rng = np.random.RandomState(0)
    plt.subplot(121)
    i = 0
    for marker in ['o', 'x', 'v', '^', '<', '>', 's', 'd']:
        plt.plot(value[i][1], value[i][0], marker, label=title[i], markersize=18)
        i += 1
    plt.legend(numpoints=1, loc='lower right', fontsize=14)
    plt.gca()
    plt.ylim([48, 65])
    plt.grid()
    plt.ylabel('Accuracy', fontsize=15)
    plt.xlabel('#Params', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.subplot(122)
    i = 0
    for marker in ['o', 'x', 'v', '^', '<', '>', 's', 'd']:
        plt.plot(value[i][2], value[i][0], marker, label=title[i], markersize=18)
        i += 1
    plt.legend(numpoints=1, loc='upper left', fontsize=14)
    plt.ylim([48,70])
    plt.gca()
    plt.grid()
    plt.ylabel('Accuracy', fontsize=15)
    plt.xlabel('#GFLOPs', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.show()

# UCF101()
hmdb51()