from math import ceil
from keras import Model
from keras.layers import Conv3D, BatchNormalization, Activation, MaxPooling3D, Dense, TimeDistributed, GlobalAveragePooling2D, Flatten
from keras.layers import GlobalMaxPooling2D, Input, Add, Lambda, Reshape, multiply
from keras.regularizers import l2
import keras.backend as K

DIM1_AXIS = 1
DIM2_AXIS = 2
DIM3_AXIS = 3
CHANNEL_AXIS = 4

def temporal_pooling(input_layer, filters_se, name):
    _, t, _, _, c = input_layer.shape.as_list()
    temporal_x = Lambda(lambda x: K.mean(x, axis=1, keepdims=True), name=name + 'temporal_pooling')(input_layer)
    x = Conv3D(filters_se, kernel_size=(1,1,1), strides=(1,1,1), padding='same', activation='relu',
               name=name + 'se_reduce_temporal')(temporal_x)
    x = Conv3D(c, kernel_size=(1,1,1), strides=(1,1,1), padding='same', activation='sigmoid',
               name=name + 'se_expand_temporal')(x)
    return x

def spatial_pooling(input_layer, filters_se, name):
    _, t, _, _, c = input_layer.shape.as_list()
    spatial_x = TimeDistributed(GlobalAveragePooling2D(), name=name + 'spatial_pooling')(input_layer)
    spatial_x = Reshape(target_shape=(t, 1, 1, c), name=name + 'spatial_pooling_reshape')(spatial_x)
    x = Conv3D(filters_se, kernel_size=(1,1,1), strides=(1,1,1), padding='same', activation='relu',
               name=name + 'se_reduce_spatial')(spatial_x)
    x = Conv3D(c, kernel_size=(1,1,1), strides=(1,1,1), padding='same', activation='sigmoid',
               name=name + 'se_expand_spatial')(x)
    return x

def _shortcut3d(input, residual, name=''):
    """3D shortcut to match input and residual and merges them with "sum"."""
    stride_dim1 = ceil(input._keras_shape[DIM1_AXIS] \
        / residual._keras_shape[DIM1_AXIS])
    stride_dim2 = ceil(input._keras_shape[DIM2_AXIS] \
        / residual._keras_shape[DIM2_AXIS])
    stride_dim3 = ceil(input._keras_shape[DIM3_AXIS] \
        / residual._keras_shape[DIM3_AXIS])
    equal_channels = residual._keras_shape[CHANNEL_AXIS] \
        == input._keras_shape[CHANNEL_AXIS]

    shortcut = input
    if stride_dim1 > 1 or stride_dim2 > 1 or stride_dim3 > 1 \
            or not equal_channels:
        shortcut = Conv3D(
            filters=residual._keras_shape[CHANNEL_AXIS],
            kernel_size=(1, 1, 1),
            strides=(stride_dim1, stride_dim2, stride_dim3),
            padding="valid",
            kernel_regularizer=None,
            name=name+'Conv_extend'
            )(input)
    return Add(name=name+'Add')([shortcut, residual])

def basic_block(input_tensor, filters, strides=1, kernel_regularizer=l2(1e-4), bn_axis=-1, name=''):
    # Convolution 3D phase
    filters_se = filters // 4
    # Conv 3x3x3
    x = Conv3D(filters, kernel_size=(3, 3, 3), strides=(strides, strides, strides), padding='same',
               use_bias=False, kernel_regularizer=kernel_regularizer,
               name=name + '3x3x3conv1')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=name + '3x3x3bn1')(x)
    x = Activation('relu', name=name + '3x3x3act1')(x)

    # Convolution (2+1)D phase
    # Conv 1x3x3
    x = Conv3D(filters, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding='same',
               kernel_regularizer=kernel_regularizer,
               use_bias=False, name=name + '1x3x3conv1')(x)
    x = BatchNormalization(axis=bn_axis, name=name + '1x3x3bn1')(x)
    x = Activation('relu', name=name + '1x3x3act1')(x)

    # Squeeze and Excitation phase for spatial
    spatial_x = spatial_pooling(x, filters_se, name=name)
    x = multiply([x, spatial_x], name=name + 'multiply_se_spatial')

    # Conv 3x1x1
    x = Conv3D(filters, kernel_size=(3, 1, 1), strides=(1, 1, 1), padding='same',
               kernel_regularizer=kernel_regularizer,
               use_bias=False, name=name + '3x1x1conv2')(x)
    x = BatchNormalization(axis=bn_axis, name=name + '3x1x1bn2')(x)
    x = Activation('relu', name=name + '3x1x1act2')(x)

    # Squeeze and Excitation phase for temporal
    temporal_x = temporal_pooling(x, filters_se, name=name)
    x = multiply([x, temporal_x], name=name + 'multiply_se_temporal')

    x = _shortcut3d(input_tensor, x, name=name)
    return x


def _residual_block3d(input_tensor, filters, kernel_regularizer, repetitions, bn_axis=-1, is_first_layer=False, name=''):
    for i in range(repetitions):
        strides = 1
        name1 = '_repeat_{}_'.format(i + 1)
        name = name + name1
        if i == 0 and not is_first_layer:
            strides = 2
        x = basic_block(input_tensor, filters=filters, strides=strides, kernel_regularizer=kernel_regularizer, bn_axis=bn_axis, name=name)
    return x


def Resnet_2p1D_18(input_shape, num_classes, repetitions=[2,2,2,2], pooling = 'avg', bn_axis=-1, kernel_regularizer=l2(1e-4)):
    input = Input(shape=input_shape, name='Input_resnet3d')
    filters = 64

    x = Conv3D(filters=filters, kernel_size=(3,7,7), strides=(1,2,2), padding='same',
               kernel_regularizer=kernel_regularizer, name='Conv1')(input)
    x = BatchNormalization(name='bn1')(x)
    x = Activation('relu', name='relu1')(x)
    if input_shape[1] == 224:
        x = MaxPooling3D(pool_size=(3,3,3), strides=(1,2,2), padding='same', name='maxpool1')(x)


    for i, r in enumerate(repetitions):
        for j in range(r):
            name = 'block_{}_repeat_{}_'.format(i + 1, j + 1)
            strides = 1
            if j == 0 and i>0:
                strides = 2
            x = basic_block(x, filters=filters, strides=strides, kernel_regularizer=kernel_regularizer,
                            bn_axis=bn_axis, name=name)
        filters = filters * 2

    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation("relu")(x)

    if pooling == 'avg':
        x = TimeDistributed(GlobalAveragePooling2D(), name='TimeDistributed_AVE')(x)
        x = Flatten(name='Flatten_AVE')(x)
    elif pooling == 'max':
        x = TimeDistributed(GlobalMaxPooling2D(), name='TimeDistributed_MAX')(x)
        x = Flatten(name='Flatten_MAX')(x)
    else:
        x = Flatten(name='Flatten_ALL')(x)

    out = Dense(units=num_classes, activation="softmax")(x)
    return Model(inputs=input, outputs=out, name="ResNet2p1D_18")


# model = Resnet_2p1D_18((16,224,224,3), 10, repetitions=[2,2,2,2])
# model.summary(line_length=150)