from math import ceil
from keras import Model
from keras.models import load_model
from keras.layers import BatchNormalization, Activation, Conv3D, Input, Flatten, multiply, Conv3DTranspose, add
from keras.layers import Add, MaxPooling3D, Dense, TimeDistributed, GlobalAveragePooling2D, GlobalMaxPooling2D, Lambda, Reshape
from keras.regularizers import l2
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

def temporal_pooling(input_layer, filters_se, name, weight_decay=1e-4):
    _, t, _, _, c = input_layer.shape.as_list()
    temporal_x = Lambda(lambda x: K.mean(x, axis=1, keepdims=True), name=name + 'temporal_pooling')(input_layer)
    x = Conv3D(filters_se, kernel_size=(1,1,1), strides=(1,1,1), padding='same', activation=swish,
               kernel_initializer=CONV_KERNEL_INITIALIZER,
               kernel_regularizer=l2(weight_decay),
               name=name + 'se_reduce_temporal')(temporal_x)
    x = Conv3D(c, kernel_size=(1,1,1), strides=(1,1,1), padding='same', activation='sigmoid',
               kernel_initializer=CONV_KERNEL_INITIALIZER,
               kernel_regularizer=l2(weight_decay),
               name=name + 'se_expand_temporal')(x)
    return x

def spatial_pooling(input_layer, filters_se, name, weight_decay=1e-4):
    _, t, _, _, c = input_layer.shape.as_list()
    spatial_x = TimeDistributed(GlobalAveragePooling2D(), name=name + 'spatial_pooling')(input_layer)
    spatial_x = Reshape(target_shape=(t, 1, 1, c), name=name + 'spatial_pooling_reshape')(spatial_x)
    x = Conv3D(filters_se, kernel_size=(1,1,1), strides=(1,1,1), padding='same', activation=swish,
               kernel_initializer=CONV_KERNEL_INITIALIZER,
               kernel_regularizer=l2(weight_decay),
               name=name + 'se_reduce_spatial')(spatial_x)
    x = Conv3D(c, kernel_size=(1,1,1), strides=(1,1,1), padding='same', activation='sigmoid',
               kernel_initializer=CONV_KERNEL_INITIALIZER,
               kernel_regularizer=l2(weight_decay),
               name=name + 'se_expand_spatial')(x)
    return x


def efficient_block(inputs, name='', bn_axis=-1, filters_in=32, filters_out=16, strides=1, weight_decay=1e-4):
    # Project phase
    filters = filters_in // 4
    filters_se = filters // 4

    x = inputs

    # Convolution (2+1)D phase
    # Conv 1x3x3
    x = Conv3D(filters, kernel_size=(1, 3, 3), strides=(1, strides, strides), padding='same',
               kernel_initializer=CONV_KERNEL_INITIALIZER,
               kernel_regularizer=l2(weight_decay),
               use_bias=False, name=name + '1x3x3conv1')(x)
    x = BatchNormalization(axis=bn_axis, name=name + '1x3x3bn1')(x)
    x = Activation(swish, name=name + '1x3x3act1')(x)

    # Squeeze and Excitation phase for spatial
    spatial_x = spatial_pooling(x, filters_se, name=name)
    x = multiply([x, spatial_x], name=name + 'multiply_se_spatial')

    # Conv 3x1x1
    x = Conv3D(filters, kernel_size=(3, 1, 1), strides=(strides, 1, 1), padding='same',
               kernel_initializer=CONV_KERNEL_INITIALIZER,
               kernel_regularizer=l2(weight_decay),
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
               kernel_regularizer=l2(weight_decay),
               name=name + 'expand_conv')(x)
    x = BatchNormalization(axis=bn_axis, name=name + 'expand_bn')(x)
    if (strides == 1 and filters_in == filters_out):
        x = add([x, inputs], name=name + 'add')
    # x = Activation(swish, name=name + 'add_activation')(x)
    return x


def EfficientNet(input_shape, num_classes, filters, repeat, strides_list, pooling = 'avg', model_name='', weight_decay=1e-4):
    input = Input(shape=input_shape, name='Input_encoder')
    bn_axis = -1

    x = Conv3D(filters=filters[0]//6, kernel_size=(3,7,7), strides=(1,2,2), padding='same',
               kernel_initializer=CONV_KERNEL_INITIALIZER,
               kernel_regularizer=l2(weight_decay),
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

# def EfficientB0Net(input_shape, num_classes, pooling = 'avg'):
#     filters = [192, 96, 144, 240, 480, 672, 1152, 1920]
#     # repeat = [1, 2, 2, 3, 3, 4, 1]
#     repeat = [2, 3, 3, 4, 4, 5, 2]
#     strides_list = [1, 2, 2, 1, 2, 1]
#     return EfficientNet(input_shape,num_classes,filters,repeat,strides_list,pooling,model_name='EfficientB0Net')
#
# def EfficientB1Net(input_shape, num_classes, pooling = 'avg'):
#     filters = [192, 96, 144, 240, 480, 672, 1152, 1920]
#     repeat = [1, 2, 2, 3, 3, 4, 1]
#     strides_list = [1, 2, 2, 1, 2, 1]
#     return EfficientNet(input_shape, num_classes, filters, repeat, strides_list, pooling, model_name='EfficientB1Net')
#
# def EfficientB2Net(input_shape, num_classes, pooling = 'avg'):
#     filters = [192, 96, 144, 288, 528, 720, 1248, 2112]
#     repeat = [2, 3, 3, 4, 4, 5, 2]
#     strides_list = [1, 2, 2, 1, 2, 1]
#     return EfficientNet(input_shape, num_classes, filters, repeat, strides_list, pooling, model_name='EfficientB2Net')

def EfficientB3Net(input_shape, num_classes, pooling = 'avg'):
    # filters = [240, 144, 192, 288, 576, 816]
    # filters = [288, 144, 192, 336, 672, 960, 1632]
    # filters = [192, 96, 144, 288, 528, 720, 1248]
    # filters = [192, 96, 144, 240, 480, 672, 1024]
    filters = [192, 96, 144, 288, 528, 720, 1024]
    repeat = [2, 3, 4, 5, 6, 1]
    strides_list = [1, 2, 2, 1, 2, 1]
    return EfficientNet(input_shape, num_classes, filters, repeat, strides_list, pooling, model_name='EfficientB3Net')

# def EfficientB4Net(input_shape, num_classes, pooling = 'avg'):
#     filters = [288, 144, 192, 336, 672, 960, 1632, 2688]
#     repeat = [2, 4, 4, 6, 8, 2]
#     strides_list = [1, 2, 2, 1, 2, 1]
#     return EfficientNet(input_shape, num_classes, filters, repeat, strides_list, pooling, model_name='EfficientB4Net')


def encoder_decoder_model(input_shape, weight_decay=1e-4):
#     filters = [240, 144, 192, 288, 576, 816, 1392]
#     repeat = [2, 3, 4, 5, 6, 2]
#     strides_list = [1, 2, 2, 1, 2, 1]
    filters = [192, 96, 144, 240, 480, 672, 1024]
    repeat = [2, 3, 4, 5, 6, 1]
    strides_list = [1, 2, 2, 1, 2, 1]

    input = Input(shape=input_shape, name='Input_encoder')
    bn_axis = -1

    x = Conv3D(filters=filters[0] // 6, kernel_size=(3, 7, 7), strides=(1, 2, 2), padding='same',
               kernel_initializer=CONV_KERNEL_INITIALIZER,
                kernel_regularizer=l2(weight_decay),
               use_bias=False, name='conv1')(input)
    x = BatchNormalization(axis=bn_axis, name='bn1')(x)
    x = Activation(swish, name='swish1')(x)
    if input_shape[1]==224:
        x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same', name='maxpool1')(x)

    x_list = []
    for i, r in enumerate(repeat):
        for j in range(r):
            if j == 0:
                filters_in = filters[i]
                filters_out = filters[i + 1]
                strides = strides_list[i]
            else:
                filters_in = filters[i + 1]
                filters_out = filters[i + 1]
                strides = 1
            name = 'block_{}_{}_'.format(i + 1, j + 1)
            x = efficient_block(x, name=name, filters_in=filters_in, filters_out=filters_out, strides=strides)
        x_list.append(x)

    #-------Deconder
    x = Conv3D(filters[-2], 1, strides=1,padding='same', use_bias=False, name='De_conv1',
               kernel_initializer=CONV_KERNEL_INITIALIZER,
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name='De_bn_conv1')(x)
    x = Activation(swish, name='De_activation_conv1')(x)
    x = Add(name='De_En_add1')([x, x_list[-2]])
    x = Conv3D(filters[-2] //4, 1, strides=1,padding='same', use_bias=False, name='De_conv2',
               kernel_regularizer=l2(weight_decay),
               kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = BatchNormalization(axis=bn_axis, name='De_bn_conv2')(x)
    x = Activation(swish, name='De_activation_conv2')(x)

    # Upsample 1
    x = Conv3DTranspose(filters[-3],3, strides=2,padding='same', use_bias=False, name='De_Upsame_1',
                        kernel_regularizer=l2(weight_decay),
                        kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = BatchNormalization(axis=bn_axis, name='De_bn_Upsame1')(x)
    x = Activation(swish, name='De_activation_Upsame1')(x)
    x = Add(name='De_En_add2')([x, x_list[-3]])

    x = Conv3D(filters[-4], 1, strides=1, padding='same', use_bias=False, name='De_conv3',
               kernel_regularizer=l2(weight_decay),
               kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = BatchNormalization(axis=bn_axis, name='De_bn_conv3')(x)
    x = Activation(swish, name='De_activation_conv3')(x)
    x = Add(name='De_En_add3')([x, x_list[-4]])
    x = Conv3D(filters[-4] // 4, 1, strides=1, padding='same', use_bias=False, name='De_conv4',
               kernel_regularizer=l2(weight_decay),
               kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = BatchNormalization(axis=bn_axis, name='De_bn_conv4')(x)
    x = Activation(swish, name='De_activation_conv4')(x)

    # Upsample 2
    x = Conv3DTranspose(filters[-5], 3, strides=2, padding='same', use_bias=False, name='De_Upsame_2',
                        kernel_regularizer=l2(weight_decay),
                        kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = BatchNormalization(axis=bn_axis, name='De_bn_Upsame2')(x)
    x = Activation(swish, name='De_activation_Upsame2')(x)
    x = Add(name='De_En_add4')([x, x_list[-5]])
    x = Conv3D(filters[-5] // 4, 1, strides=1, padding='same', use_bias=False, name='De_conv5',
               kernel_regularizer=l2(weight_decay),
               kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = BatchNormalization(axis=bn_axis, name='De_bn_conv5')(x)
    x = Activation(swish, name='De_activation_conv5')(x)

    # Upsample 3
    x = Conv3DTranspose(filters[-6], 3, strides=2, padding='same', use_bias=False, name='De_Upsame_3',
                        kernel_regularizer=l2(weight_decay),
                        kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = BatchNormalization(axis=bn_axis, name='De_bn_Upsame3')(x)
    x = Activation(swish, name='De_activation_Upsame3')(x)
    x = Add(name='De_En_add5')([x, x_list[-6]])

    # Upsample 4
    x = Conv3D(filters[0]//4, 1, strides=1, padding='same', use_bias=False, name='De_conv6',
               kernel_regularizer=l2(weight_decay),
               kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = BatchNormalization(axis=bn_axis, name='De_bn_conv6')(x)
    x = Activation(swish, name='De_activation_conv6')(x)
    x = Conv3DTranspose(filters[0]//6, 7, strides=(2,4,4), padding='same', use_bias=False, name='De_Upsame_4',
                        kernel_regularizer=l2(weight_decay),
                        kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = BatchNormalization(axis=bn_axis, name='De_bn_Upsame4')(x)
    x = Activation(swish, name='De_activation_Upsame4')(x)
    out = Conv3D(3, 1, strides=1, padding='same', name='out_predict', kernel_regularizer=l2(weight_decay),
                 kernel_initializer=CONV_KERNEL_INITIALIZER, activation='tanh')(x)
    # x = BatchNormalization(axis=bn_axis, name='De_bn_conv7')(x)
    # out = Activation('tanh', name='act_output')(x)
    return Model(inputs=input, outputs=out, name="encoder_decoder")

def load_weight(ed_model, effcient_model_path):
    en_model = load_model(effcient_model_path)
    print("Load weight path: ", effcient_model_path)
    for i in range(398):
        print('Load weight layer: ', i)
        ed_model.layers[i].set_weights(en_model.layers[i].get_weights())
        ed_model.layers[i].trainable = False
    print('Load weight completed')
    return ed_model

# efficientB0 = EfficientB3Net(input_shape=(16,224,224,3), num_classes=400)
# efficientB0.summary(line_length=150)

# model = encoder_decoder_model((16,224,224,3))
# model.summary(line_length=150)
# print(len(model.layers))
# print(model.layers[397].get_config())

# effcient_model_path = 'saved_models/EfficientNet3D/EfficientNet3D_.hdf5'
# model = load_weight(model,effcient_model_path)
# print('COMPLETED')

# en_model = load_model('saved_models/EfficientNet.hdf5')
# en_model = en_model.layers[-2]
# print(len(en_model.layers))
# en_model.summary(line_length=150)
# for i in range(454, 460):
#     print(i)
#     model.layers[i].set_weights(en_model.layers[i].get_weights())





