import numpy as np
from keras import Model
from keras.layers import Flatten, Dense, BatchNormalization, Activation, TimeDistributed, GlobalAveragePooling2D
from keras.layers import GlobalAveragePooling3D, Concatenate
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
import keras.backend as K

# from C_data_generator import *
from C1_data_generator import *

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

def rebuild_model(model_path, nb_classes):
    old_model = load_model(model_path)
    for i in range(len(old_model.layers)):
        old_model.layers[i].trainable = True

    x14 = old_model.layers[-1].output
    x7 = old_model.layers[-2].output

    x14 = GlobalAveragePooling3D()(x14)
    x7 = TimeDistributed(GlobalAveragePooling2D())(x7)
    x7 = Flatten()(x7)
    x = Concatenate()([x7, x14])
    out = Dense(nb_classes, activation='softmax')(x)
    new_model = Model(inputs=old_model.input, outputs=out)
    return new_model

def read_resnet_model(model_path):
    model = load_model(model_path)
    return model

def train(frames_per_clip, model_path, image_shape=None, batch_size=32, nb_epoch=100):
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

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = len(train_data) // batch_size
    val_steps_per_epoch = len(test_data) // batch_size

    with tf.device("/gpu:0"):
        new_model = rebuild_model(model_path, nb_classes=len(classes))
        # new_model = read_resnet_model(model_path)
    for i in range(len(new_model.layers)):
        new_model.layers[i].trainable = True

    new_model.summary(line_length=150)
    sgd = SGD(lr=0.01, momentum=0.99, nesterov=True)
    new_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])

    model_name = 'new_model.hdf5'
    checkpointer = ModelCheckpoint(filepath=model_name, verbose=2, save_best_only=True, monitor='val_acc', mode='max')
    reduce_lr = ReduceLROnPlateau(factor=0.1, cooldown=0, patience=10, min_lr=1e-8, verbose=2, monitor='val_loss', mode='min')
    callback = [checkpointer, reduce_lr]

    new_model.fit_generator(generator=frame_generator(batch_size, train_data, frames_per_clip, crop_size, classes, is_train=True),
                                   steps_per_epoch=steps_per_epoch, epochs=nb_epoch,
                                   verbose=1, callbacks=callback,
                                   validation_data=frame_generator(batch_size, test_data, frames_per_clip, crop_size, classes, is_train=False),
                                   validation_steps=val_steps_per_epoch, use_multiprocessing=True, workers=12)

def main():
    frame_per_clip = 16
    batch_size = 8
    nb_epoch = 10000
    image_shape = (224,224,3)
    model_path = "distilled_shuffleNet.hdf5"

    train(frame_per_clip, model_path, image_shape=image_shape, batch_size=batch_size,nb_epoch=nb_epoch)

if __name__ == '__main__':
    main()
