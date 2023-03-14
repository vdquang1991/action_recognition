import numpy as np
import random
from keras import Model
from keras.layers import Conv3D, Input, BatchNormalization, Dense, Lambda, Activation, MaxPooling3D, Concatenate, Flatten
from keras.layers import TimeDistributed, GlobalAveragePooling2D, GlobalMaxPooling2D
# from keras.regularizers import l2
import keras.backend as K

def channel_split(x, name=''):
    # equipartition
    in_channles = x.shape.as_list()[-1]
    ip = in_channles // 2
    c_hat = Lambda(lambda z: z[:, :, :, :, 0:ip], name='%s/sp%d_slice' % (name, 0))(x)
    c = Lambda(lambda z: z[:, :, :, :, ip:], name='%s/sp%d_slice' % (name, 1))(x)
    return c_hat, c


def channel_shuffle(x):
    # batch_size, time, height, width, channels = x.shape.as_list()
    # x = K.stack(x, axis=4)
    # x = K.permute_dimensions(x, (0, 1, 2, 3, 5, 4))
    # x = K.reshape(x, [batch_size, time, height, width, channels])

    batch_size, time, height, width, channels = x.shape.as_list()
    channels_per_split = channels // 2
    x = K.reshape(x, [-1, time, height, width, 2, channels_per_split])
    x = K.permute_dimensions(x, (0,1,2,3,5,4))
    x = K.reshape(x, [-1, time, height, width, channels])
    return x

def shuffle_unit(inputs, out_channels, bottleneck_ratio,strides=2,stage=1,block=1, weight_decay=5e-4):
    bottleneck_channels = int(out_channels * bottleneck_ratio)
    prefix = 'stage{}/block{}'.format(stage, block)

    if strides < 2:
        c_hat, c = channel_split(inputs, '{}/spl'.format(prefix))
        inputs = c

    x = Conv3D(bottleneck_channels, kernel_size=(1, 3, 3), strides=(1, strides, strides), padding='same',
               use_bias=False, name='{}/1x3x3conv_0'.format(prefix))(inputs)
    x = BatchNormalization(name='{}/bn_1x3x3conv_0'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_1x3x3conv_0'.format(prefix))(x)

    x = Conv3D(bottleneck_channels, kernel_size=(3,1,1), strides=(strides,1,1), padding='same',
               use_bias=False, name='{}/3x1x1conv_1'.format(prefix))(x)
    x = BatchNormalization(name='{}/bn_3x1x1conv_1'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_3x1x1conv_1'.format(prefix))(x)

    x = Conv3D(bottleneck_channels, kernel_size=(1,3,3), strides=(1,1,1), padding='same',
               use_bias=False, name='{}/1x3x3conv_2'.format(prefix))(x)
    x = BatchNormalization(name='{}/bn_1x3x3conv_2'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_1x3x3conv_2'.format(prefix))(x)

    x = Conv3D(bottleneck_channels, kernel_size=(3,1,1), strides=(1,1,1), padding='same',
               use_bias=False, name='{}/3x1x1conv_3'.format(prefix))(x)
    x = BatchNormalization(name='{}/bn_3x1x1conv_3'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_3x1x1conv_3'.format(prefix))(x)

    if strides==2:
        y = Conv3D(bottleneck_channels, kernel_size=(3, 3, 3), strides=(strides, strides, strides), padding='same',
                   use_bias=False, name='{}/1x3x3conv_plus'.format(prefix))(inputs)
        y = BatchNormalization(name='{}/bn_1x3x3conv_plus'.format(prefix))(y)
        y = Activation('relu', name='{}/relu_1x3x3conv_plus'.format(prefix))(y)

        # y = Conv3D(bottleneck_channels, kernel_size=(3, 1, 1), strides=(strides,1,1), padding='same',
        #            use_bias=False, name='{}/3x1x1conv_plus'.format(prefix))(y)
        # y = BatchNormalization(name='{}/bn_3x1x1conv_plus'.format(prefix))(y)
        # y = Activation('relu', name='{}/relu_3x1x1conv_plus'.format(prefix))(y)

        ret = Concatenate(name='{}/concat_s2'.format(prefix))([x, y])
    else:
        ret = Concatenate(name='{}/concat_s1'.format(prefix))([x, c_hat])

    ret = Lambda(channel_shuffle, name='{}/channel_shuffle'.format(prefix))(ret)
    return ret



def block(x, channel_map, bottleneck_ratio, repeat=1, stage=1):
    x = shuffle_unit(x, out_channels=channel_map[stage-1],
                      strides=2,bottleneck_ratio=bottleneck_ratio,stage=stage,block=1)
    for i in range(1, repeat+1):
        x = shuffle_unit(x, out_channels=channel_map[stage-1],strides=1,
                          bottleneck_ratio=bottleneck_ratio,stage=stage, block=(1+i))
    return x

def ShuffleNet_2p1D(input_shape, nb_classes, num_shuffle_units=[3,3,3], bottleneck_ratio=1, scale_factor=1.0, pooling = 'avg'):
    out_channels_in_stage = [64, 64, 128, 256]
    crop_shape = input_shape[1]

    inp = Input(shape=input_shape, name='inp_ShuffleNet')
    x = Conv3D(out_channels_in_stage[0], kernel_size=(3, 7, 7), strides=(1, 2, 2), padding='same', use_bias=False,
               name='stage1/3x7x7conv1')(inp)
    x = BatchNormalization(name='stage1/bn')(x)
    x = Activation('relu', name='stage1/relu')(x)

    if crop_shape == 224:
        x = MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='same', name='stage1/maxpool')(x)

    for stage in range(len(num_shuffle_units)):
        repeat = num_shuffle_units[stage]
        x = block(x, out_channels_in_stage, repeat=repeat, bottleneck_ratio=bottleneck_ratio, stage=stage + 2)

    if pooling == 'avg':
        x = TimeDistributed(GlobalAveragePooling2D(), name='TimeDistributed_AVE')(x)
        x = Flatten(name='Flatten_AVE')(x)
    elif pooling == 'max':
        x = TimeDistributed(GlobalMaxPooling2D(), name='TimeDistributed_MAX')(x)
        x = Flatten(name='Flatten_MAX')(x)
    else:
        x = Flatten(name='Flatten_ALL')(x)

    out = Dense(nb_classes, activation='softmax', name='output_layer')(x)
    model = Model(inputs=inp, outputs=out, name='ShuffleNet2p1D')
    return model

input_shape = (16,224,224,3)
nb_classes  = 400

model = ShuffleNet_2p1D(input_shape=input_shape, nb_classes=nb_classes, pooling='avg', num_shuffle_units=[4,12,4])
model.summary(line_length=150)






