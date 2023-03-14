from math import ceil
from keras import Model
from keras.layers import BatchNormalization, Activation, Conv3D, Input, Flatten, multiply
from keras.layers import Add, MaxPooling3D, Dense, TimeDistributed, GlobalAveragePooling2D, GlobalMaxPooling2D, Lambda, Reshape
from keras.regularizers import l2
import keras.backend as K

DIM1_AXIS = 1
DIM2_AXIS = 2
DIM3_AXIS = 3
CHANNEL_AXIS = 4

def temporal_pooling(input_layer, filters_se):
    _, t, _, _, c = input_layer.shape.as_list()
    temporal_x = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(input_layer)
    x = Conv3D(filters_se, kernel_size=(1,1,1), strides=(1,1,1), padding='same', activation='relu',
               )(temporal_x)
    x = Conv3D(c, kernel_size=(1,1,1), strides=(1,1,1), padding='same', activation='sigmoid',
               )(x)
    return x

def spatial_pooling(input_layer, filters_se):
    _, t, _, _, c = input_layer.shape.as_list()
    spatial_x = TimeDistributed(GlobalAveragePooling2D(), )(input_layer)
    spatial_x = Reshape(target_shape=(t, 1, 1, c))(spatial_x)
    x = Conv3D(filters_se, kernel_size=(1,1,1), strides=(1,1,1), padding='same', activation='relu',
               )(spatial_x)
    x = Conv3D(c, kernel_size=(1,1,1), strides=(1,1,1), padding='same', activation='sigmoid',
               )(x)
    return x

def _bn_relu(input):
    """Helper to build a BN -> relu block (by @raghakot)."""
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)

def _bn_relu_conv3d(**conv_params):
    """Helper to build a  BN -> relu -> conv3d block."""
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer",
                                                "glorot_uniform")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer",None)

    def f(input):
        activation = _bn_relu(input)
        return Conv3D(filters=filters, kernel_size=kernel_size,
                      strides=strides, kernel_initializer=kernel_initializer,
                      padding=padding,
                      kernel_regularizer=kernel_regularizer)(activation)
    return f

def _shortcut3d(input, residual):
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
            kernel_regularizer=None
            )(input)
    return Add()([shortcut, residual])

def basic_block(filters, strides=(1, 1, 1), kernel_regularizer=l2(1e-4),
                is_first_block_of_first_layer=False):
    """Basic 3 X 3 X 3 convolution blocks. Extended from raghakot's 2D impl."""
    def f(input):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv3D(filters=filters, kernel_size=(3, 3, 3),
                           strides=strides, padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=kernel_regularizer
                           )(input)
        else:
            conv1 = _bn_relu_conv3d(filters=filters,
                                    kernel_size=(3, 3, 3),
                                    strides=strides,
                                    kernel_regularizer=kernel_regularizer
                                    )(input)

        residual = _bn_relu_conv3d(filters=filters, kernel_size=(3, 3, 3),
                                   kernel_regularizer=kernel_regularizer
                                   )(conv1)
        return _shortcut3d(input, residual)

    return f

def bottleneck(filters, strides=(1, 1, 1), kernel_regularizer=l2(1e-4),
               is_first_block_of_first_layer=False):
    """Basic 3 X 3 X 3 convolution blocks. Extended from raghakot's 2D impl."""
    def f(input):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv3D(filters=filters, kernel_size=(1, 1, 1),
                              strides=strides, padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=kernel_regularizer
                              )(input)
        else:
            conv_1_1 = _bn_relu_conv3d(filters=filters, kernel_size=(1, 1, 1),
                                       strides=strides,
                                       kernel_regularizer=kernel_regularizer
                                       )(input)

        conv_3_3 = _bn_relu_conv3d(filters=filters, kernel_size=(3, 3, 3),
                                   kernel_regularizer=kernel_regularizer
                                   )(conv_1_1)
        residual = _bn_relu_conv3d(filters=filters * 4, kernel_size=(1, 1, 1),
                                   kernel_regularizer=kernel_regularizer
                                   )(conv_3_3)

        return _shortcut3d(input, residual)

    return f



def _residual_block3d(filters, kernel_regularizer, repetitions,is_first_layer=False):
    def f(input):
        for i in range(repetitions):
            strides = (1, 1, 1)
            if i == 0 and not is_first_layer:
                strides = (2, 2, 2)
            input = basic_block(filters=filters, strides=strides, kernel_regularizer=kernel_regularizer,
                               is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input
    return f


def Resnet_3D(input_shape, num_classes, repetitions, pooling = 'avg', kernel_regularizer=l2(1e-4)):
    input = Input(shape=input_shape, name='Input_resnet3d')

    filters = 64

    x = Conv3D(filters=filters, kernel_size=(7,7,7), strides=(1,2,2), padding='same',
               kernel_regularizer=kernel_regularizer, name='Conv1')(input)
    x = BatchNormalization(name='bn1')(x)
    x = Activation('relu', name='relu1')(x)
    if input_shape[1] == 224:
        x = MaxPooling3D(pool_size=(3,3,3), strides=(1,2,2), padding='same', name='maxpool1')(x)

    for i, r in enumerate(repetitions):
        x = _residual_block3d(filters=filters, kernel_regularizer=kernel_regularizer,repetitions=r, is_first_layer=(i == 0))(x)
        filters *= 2

    x = _bn_relu(x)

    if pooling == 'avg':
        x = TimeDistributed(GlobalAveragePooling2D(), name='TimeDistributed_AVE')(x)
        x = Flatten(name='Flatten_AVE')(x)
    elif pooling == 'max':
        x = TimeDistributed(GlobalMaxPooling2D(), name='TimeDistributed_MAX')(x)
        x = Flatten(name='Flatten_MAX')(x)
    else:
        x = Flatten(name='Flatten_ALL')(x)

    out = Dense(units=num_classes, activation="softmax")(x)
    return Model(inputs=input, outputs=out, name="ResNet3D")



# resnet_50 = Resnet_3D(input_shape=(16,224,224,3), num_classes=400, repetitions=[3,4,6,3])
# resnet_50.summary(line_length=150)


# resnet18_3D = Resnet_3D((16,112,112,3), 400, repetitions=[2,2,2,2])
# resnet18_3D.summary(line_length=150)



