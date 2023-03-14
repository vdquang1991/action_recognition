from keras import Model
from keras.models import load_model
from keras.layers import Input, Conv3D, Activation, MaxPooling3D, Lambda, multiply, Dense, add, TimeDistributed, Reshape
from keras.layers import GlobalAveragePooling2D, Flatten, GlobalMaxPooling2D
import keras.backend as K
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint

from callbacks.BN16 import BatchNormalizationF16
from callbacks.epochcheckpoint import EpochCheckpoint
from callbacks.trainingmonitor import TrainingMonitor

from C_data_generator import *

K.set_floatx('float16')
K.set_epsilon(1e-4)

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

    # Convolution (2+1)D phase
    # Conv 1x3x3
    x = Conv3D(filters, kernel_size=(1, 3, 3), strides=(1, strides, strides), padding='same',
               kernel_initializer=CONV_KERNEL_INITIALIZER,
               use_bias=False, name=name + '1x3x3conv1')(x)
    x = BatchNormalizationF16(axis=bn_axis, name=name + '1x3x3bn1')(x)
    x = Activation(swish, name=name + '1x3x3act1')(x)

    # Squeeze and Excitation phase for spatial
    spatial_x = spatial_pooling(x, filters_se, name=name)
    x = multiply([x, spatial_x], name=name + 'multiply_se_spatial')

    # Conv 3x1x1
    x = Conv3D(filters, kernel_size=(3, 1, 1), strides=(strides, 1, 1), padding='same',
               kernel_initializer=CONV_KERNEL_INITIALIZER,
               use_bias=False, name=name + '3x1x1conv2')(x)
    x = BatchNormalizationF16(axis=bn_axis, name=name + '3x1x1bn2')(x)
    x = Activation(swish, name=name + '3x1x1act2')(x)

    # Squeeze and Excitation phase for temporal
    temporal_x = temporal_pooling(x, filters_se, name=name)
    x = multiply([x, temporal_x], name=name + 'multiply_se_temporal')

    # Expand phase
    x = Conv3D(filters_out, 1,
               padding='same',
               use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER,
               name=name + 'expand_conv')(x)
    x = BatchNormalizationF16(axis=bn_axis, name=name + 'expand_bn')(x)
    if (strides == 1 and filters_in == filters_out):
        x = add([x, inputs], name=name + 'add')
    x = Activation(swish, name=name + 'add_activation')(x)
    return x


def EfficientNet(input_shape, num_classes, filters, repeat, strides_list, pooling = 'avg', model_name=''):
    input = Input(shape=input_shape, name='Input_encoder')
    bn_axis = -1

    x = Conv3D(filters=filters[0]//6, kernel_size=(3,7,7), strides=(1,2,2), padding='same',
               kernel_initializer=CONV_KERNEL_INITIALIZER,
               use_bias=False, name='conv1')(input)
    x = BatchNormalizationF16(axis=bn_axis, name='bn1')(x)
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
    # filters = [240, 144, 192, 288, 576, 816]
    filters = [288, 144, 192, 336, 672, 960, 1632]
    # filters = [192, 96, 144, 288, 528, 720, 1248]
    # filters = [192, 96, 144, 240, 480, 672, 1024]
    # filters = [192, 96, 144, 288, 528, 720, 1024]
    repeat = [2, 3, 4, 5, 6, 1]
    strides_list = [1, 2, 2, 1, 2, 1]
    return EfficientNet(input_shape, num_classes, filters, repeat, strides_list, pooling, model_name='EfficientB3Net')

def train(frames_per_clip, image_shape=None, batch_size=32, nb_epoch=100, model_name="ShuffleNet2p1D", repetitions=None,
          begin_training=True, start_epoch=0, num_GPU=1):
    crop_size = image_shape[0]

    train_data = get_data('train.csv')
    test_data = get_data('test.csv')

    classes = get_classes(train_data)
    print('Number of classes:', len(classes))
    print('Train set:', len(train_data))
    print('Test set:', len(test_data))

    train_data = clean_data(train_data, 8, classes, 3000)
    test_data = clean_data(test_data, frames_per_clip, classes, 3000)
    print('Train set after clean:', len(train_data))
    print('Test set after clean:', len(test_data))

    total_data = train_data + test_data
    # random.shuffle(test_data)
    # test_data = test_data[:1000]
    print('Total set after clean:', len(total_data))
    print('Test set after clean:', len(test_data))

    if not os.path.exists(model_name):
        os.mkdir(model_name)
    if not os.path.exists(os.path.join(model_name, "output")):
        os.mkdir(os.path.join(model_name, "output"))
    if not os.path.exists(os.path.join(model_name, "checkpoints")):
        os.mkdir(os.path.join(model_name, "checkpoints"))

    steps_per_epoch = len(total_data) // batch_size
    val_steps_per_epoch = len(test_data) // batch_size

    total = ""
    if repetitions is not None:
        for value in repetitions:
            total = total + str(value) + "_"

    filepath = os.path.join(model_name, model_name + "_" + str(total) + ".hdf5")
    checkpoint_path = os.path.join(model_name, 'checkpoints')
    plotPath = os.path.join(model_name, "output")
    jsonPath = os.path.join(model_name, "output")
    jsonName = model_name + '_result.json'

    sgd = SGD(lr=0.0001, momentum=0.99, nesterov=True)
    early_stopper = EarlyStopping(patience=200, mode='min', verbose=2, monitor='val_loss')
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=2, save_best_only=True, monitor='val_acc', mode='max')
    reduce_lr = ReduceLROnPlateau(factor=0.1, cooldown=0, patience=10, min_lr=1e-8, verbose=2, monitor='val_loss',
                                  mode='min')
    log_path = os.path.join(model_name, "logs")
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    tensorboard = TensorBoard(log_dir=log_path)

    if num_GPU > 1:
        epoch_checkpoint = EpochCheckpoint(checkpoint_path, every=1, startAt=start_epoch, multi_GPU=True)
    else:
        epoch_checkpoint = EpochCheckpoint(checkpoint_path, every=1, startAt=start_epoch, multi_GPU=False)
    callback = [early_stopper, checkpointer, reduce_lr, tensorboard,
                epoch_checkpoint,
                TrainingMonitor(plotPath, jsonPath=jsonPath, jsonName=jsonName, startAt=start_epoch)]

    if begin_training:
        model = EfficientB3Net(input_shape=(frames_per_clip, crop_size, crop_size, 3), num_classes=len(classes))
        model.summary(line_length=150)
    else:
        modelpath = os.path.join(model_name, 'checkpoints', 'epoch_' + str(start_epoch) + '.hdf5')
        print("LOAD MODEL: ", model_name, modelpath)
        model = load_model(modelpath)
        print("LOAD MODEL COMPLETED")
        # K.set_value(model.optimizer.lr, 0.001)
        model.summary(line_length=150)


    model.compile(optimizer=sgd, loss='categorical_crossentropy',
                            metrics=['accuracy', 'top_k_categorical_accuracy'])
    print(K.eval(model.optimizer.lr))
    hist = model.fit_generator(
        generator=frame_generator(batch_size, total_data, frames_per_clip, crop_size, classes, is_train=True),
        steps_per_epoch=steps_per_epoch, epochs=nb_epoch,
        verbose=1, callbacks=callback,
        validation_data=frame_generator(batch_size, test_data, frames_per_clip, crop_size, classes, is_train=False),
        validation_steps=val_steps_per_epoch, use_multiprocessing=True, workers=12)

efficient = EfficientB3Net(input_shape=(16,224,224,3), num_classes=400)
efficient.summary(line_length=150)