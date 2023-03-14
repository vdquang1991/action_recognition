from keras import Model
from keras.layers import BatchNormalization, Activation, Conv3D, Input, Flatten, multiply, add
from keras.layers import Add, MaxPooling3D, Dense, TimeDistributed, GlobalAveragePooling2D, GlobalMaxPooling2D, Lambda, Reshape
import keras.backend as K



CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

def swish(x):
    return x * K.sigmoid(x)

def temporal_pooling(input_layer, filters_se, name):
    _, t, _, _, c = input_layer.shape.as_list()
    temporal_x = Lambda(lambda x: K.mean(x, axis=1, keepdims=True), name=name + 'temporal_pooling')(input_layer)
    x = Conv3D(filters_se, kernel_size=(1,1,1), strides=(1,1,1), padding='same', activation=swish,
               kernel_initializer=CONV_KERNEL_INITIALIZER,
               name=name + 'se_reduce_temporal')(temporal_x)
    x = Conv3D(c, kernel_size=(1,1,1), strides=(1,1,1), padding='same', activation='sigmoid',
               kernel_initializer=CONV_KERNEL_INITIALIZER,
               name=name + 'se_expand_temporal')(x)
    return x

def spatial_pooling(input_layer, filters_se, name):
    _, t, _, _, c = input_layer.shape.as_list()
    spatial_x = TimeDistributed(GlobalAveragePooling2D(), name=name + 'spatial_pooling')(input_layer)
    spatial_x = Reshape(target_shape=(t, 1, 1, c), name=name + 'spatial_pooling_reshape')(spatial_x)
    x = Conv3D(filters_se, kernel_size=(1,1,1), strides=(1,1,1), padding='same', activation=swish,
               kernel_initializer=CONV_KERNEL_INITIALIZER,
               name=name + 'se_reduce_spatial')(spatial_x)
    x = Conv3D(c, kernel_size=(1,1,1), strides=(1,1,1), padding='same', activation='sigmoid',
               kernel_initializer=CONV_KERNEL_INITIALIZER,
               name=name + 'se_expand_spatial')(x)
    return x

def efficient_block(inputs, name='', bn_axis=-1, filters_in=32, filters_out=16, strides=1):
    # Project phase
    filters = filters_in // 4
    filters_se = filters // 4

    x = inputs

    # Convolution 3D phase
    # Conv 3x3x3
    x = Conv3D(filters, kernel_size=(3, 3, 3), strides=(strides, strides, strides), padding='same',
               kernel_initializer=CONV_KERNEL_INITIALIZER,
               use_bias=False, name=name + '3x3x3conv1')(x)
    x = BatchNormalization(axis=bn_axis, name=name + '3x3x3bn1')(x)
    x = Activation(swish, name=name + '3x3x3act1')(x)

    # Convolution (2+1)D phase
    # Conv 1x3x3
    x = Conv3D(filters, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding='same',
               kernel_initializer=CONV_KERNEL_INITIALIZER,
               use_bias=False, name=name + '1x3x3conv1')(x)
    x = BatchNormalization(axis=bn_axis, name=name + '1x3x3bn1')(x)
    x = Activation(swish, name=name + '1x3x3act1')(x)

    # Squeeze and Excitation phase for spatial
    spatial_x = spatial_pooling(x, filters_se, name=name)
    x = multiply([x, spatial_x], name=name + 'multiply_se_spatial')

    # Conv 3x1x1
    x = Conv3D(filters, kernel_size=(3, 1, 1), strides=(1, 1, 1), padding='same',
               kernel_initializer=CONV_KERNEL_INITIALIZER,
               use_bias=False, name=name + '3x1x1conv2')(x)
    x = BatchNormalization(axis=bn_axis, name=name + '3x1x1bn2')(x)
    x = Activation(swish, name=name + '3x1x1act2')(x)

    # Squeeze and Excitation phase for temporal
    temporal_x = temporal_pooling(x, filters_se, name=name)
    x = multiply([x, temporal_x], name=name + 'multiply_se_temporal')

    # Expand phase
    x = Conv3D(filters_out, 1,
               padding='same',
               use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER,
               name=name + 'expand_conv')(x)
    x = BatchNormalization(axis=bn_axis, name=name + 'expand_bn')(x)
    if (strides == 1 and filters_in == filters_out):
        x = add([x, inputs], name=name + 'add')
    # x = Activation(swish, name=name + 'add_activation')(x)
    return x


def EfficientNet(input_shape, num_classes, filters, repeat, strides_list, pooling = 'avg', model_name=''):
    input = Input(shape=input_shape, name='Input_layer')
    bn_axis = -1

    x = Conv3D(filters=filters[0]//4, kernel_size=(3,7,7), strides=(1,2,2), padding='same',
               kernel_initializer=CONV_KERNEL_INITIALIZER,
               use_bias=False, name='conv1')(input)
    x = BatchNormalization(axis=bn_axis, name='bn1')(x)
    x = Activation(swish, name='swish1')(x)
    if input_shape[1]==224:
        x = MaxPooling3D(pool_size=(3,3,3), strides=(1,2,2), padding='same', name='maxpool1')(x)

    for i, r in enumerate(repeat):
        for j in range(r):
            if j==0:
                filters_in = filters[i]
                filters_out = filters[i + 1]
                strides = strides_list[i]
            else:
                filters_in = filters[i + 1]
                filters_out = filters[i + 1]
                strides = 1
            name = 'block_{}_repeat_{}_'.format(i+1, j+1)
            x = efficient_block(x, name=name, filters_in=filters_in, filters_out=filters_out, strides=strides)

    if pooling == 'avg':
        x = TimeDistributed(GlobalAveragePooling2D(), name='TimeDistributed_AVE')(x)
        x = Flatten(name='Flatten_AVE')(x)
    elif pooling == 'max':
        x = TimeDistributed(GlobalMaxPooling2D(), name='TimeDistributed_MAX')(x)
        x = Flatten(name='Flatten_MAX')(x)
    else:
        x = Flatten(name='Flatten_ALL')(x)

    out = Dense(units=num_classes, activation="softmax", kernel_initializer=DENSE_KERNEL_INITIALIZER)(x)
    return Model(inputs=input, outputs=out, name=model_name)


def EfficientB3Net(input_shape, num_classes, pooling = 'avg'):
    filters = [192, 96, 144, 288, 528, 720, 1248]
    repeat = [2, 3, 4, 5, 6, 1]
    strides_list = [1, 2, 2, 1, 2, 1]
    return EfficientNet(input_shape, num_classes, filters, repeat, strides_list, pooling, model_name='EfficientB3Net')