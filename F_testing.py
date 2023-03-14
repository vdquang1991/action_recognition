import numpy as np
import os
from keras import Model
from keras.models import load_model
from C_data_generator import *
from keras.applications import ResNet50V2, ResNet101V2
from keras.layers import Input, GlobalAveragePooling2D, Dense, BatchNormalization, Dropout, Activation
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import tensorflow as tf
import keras.backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

def resnet101(input_shape, nb_classess):
    inp = Input(shape=input_shape, name='input')
    resnet50_model = ResNet101V2(include_top=False, weights='imagenet', input_tensor=inp, input_shape=None,
                                 pooling=None)
    for layer in resnet50_model.layers:
        layer.trainable = False
    x = resnet50_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(nb_classess)(x)
    out = Activation('softmax')(x)
    model = Model(inputs=inp, outputs=out)
    return model

def testing2D(image_shape, batch_size=32, nb_epoch=100, model_name=''):
    crop_size = image_shape[0]

    train_data = get_data('train.csv')
    test_data = get_data('test.csv')

    classes = get_classes(train_data)
    print('Number of classes:', len(classes))
    print('Train set:', len(train_data))
    print('Test set:', len(test_data))

    if not os.path.exists(model_name):
        os.mkdir(model_name)

    model = resnet101(image_shape,len(classes))
    model.summary(line_length=120)

    steps_per_epoch = len(train_data) * 5 // batch_size
    val_steps_per_epoch = len(test_data) // batch_size
    sgd = SGD(lr=0.001, momentum=0.99, nesterov=True)

    filepath = os.path.join(model_name, model_name + '.hdf5')
    early_stopper = EarlyStopping(patience=50, mode='min', verbose=2, monitor='val_loss')
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=2, save_best_only=True, monitor='val_acc', mode='max')
    reduce_lr = ReduceLROnPlateau(factor=0.1, cooldown=0, patience=15, min_lr=1e-8, verbose=2, monitor='val_loss',
                                  mode='min')
    callback = [early_stopper, checkpointer, reduce_lr]

    model.compile(optimizer='adadelta', loss='squared_hinge',
                  metrics=['accuracy', 'top_k_categorical_accuracy'])
    hist = model.fit_generator(
        generator=img_generator(batch_size, train_data, crop_size, classes, is_train=True),
        steps_per_epoch=steps_per_epoch, epochs=nb_epoch,
        verbose=1, callbacks=callback,
        validation_data=img_generator(batch_size, test_data, crop_size, classes, is_train=False),
        validation_steps=val_steps_per_epoch, use_multiprocessing=True, workers=12)

def testing3D(frames_per_clip, image_shape=None, batch_size=32, nb_epoch=100, model_name="ShuffleNet2p1D", mode_path=""):
    crop_size = image_shape[0]

    train_data = get_data('train.csv')
    test_data = get_data('test.csv')

    classes = get_classes(train_data)
    print('Number of classes:', len(classes))
    print('Train set:', len(train_data))
    print('Test set:', len(test_data))

    if not os.path.exists(model_name):
        os.mkdir(model_name)

    model = load_model(mode_path)
    model.summary(line_length=150)

    steps_per_epoch = len(train_data) // batch_size
    val_steps_per_epoch = len(test_data) // batch_size
    sgd = SGD(lr=0.001, momentum=0.99, nesterov=True)

    filepath = os.path.join(model_name, model_name + '.hdf5')
    early_stopper = EarlyStopping(patience=50, mode='min', verbose=2, monitor='val_loss')
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=2, save_best_only=True, monitor='val_acc', mode='max')
    reduce_lr = ReduceLROnPlateau(factor=0.1, cooldown=0, patience=15, min_lr=1e-8, verbose=2, monitor='val_loss',
                                  mode='min')
    callback = [early_stopper, checkpointer, reduce_lr]

    model.compile(optimizer=sgd, loss='categorical_crossentropy',
                  metrics=['accuracy', 'top_k_categorical_accuracy'])
    hist = model.fit_generator(
        generator=frame_generator(batch_size, train_data, crop_size, classes, is_train=True),
        steps_per_epoch=steps_per_epoch, epochs=nb_epoch,
        verbose=1, callbacks=callback,
        validation_data=frame_generator(batch_size, test_data, frames_per_clip, crop_size, classes, is_train=False),
        validation_steps=val_steps_per_epoch, use_multiprocessing=True, workers=12)

def main():
    nb_epoch = 200
    frames_per_clip = 16
    image_shape = (224, 224, 3)
    num_GPU = 1
    test_type = '3D'

    if test_type == '3D':
        batch_size = 8
        model_name = 'ResNet50_3D'
        mode_path = ''
        testing3D(frames_per_clip, image_shape,batch_size,nb_epoch=nb_epoch,model_name=model_name, mode_path=mode_path)
    else:
        model_name = "ResNet101_2D"
        batch_size = 32
        testing2D(image_shape,batch_size=batch_size,nb_epoch=nb_epoch,model_name=model_name)

if __name__ == '__main__':
    main()