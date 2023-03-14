import tensorflow as tf
import os
import keras.backend as K
from keras.models import load_model
from keras import Model
from keras.layers import Conv3D, BatchNormalization, Activation, GlobalAveragePooling2D, GlobalAveragePooling3D, TimeDistributed, Dense
from keras.layers import Flatten, Concatenate
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from C_data_generator import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

def rebuild_model(model_path, nb_classes=400):
    old_model = load_model(model_path)
    for i in range(len(old_model.layers)):
        old_model.layers[i].trainable = True

    x = old_model.layers[294].output
    x = Conv3D(1024, kernel_size=1, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling3D()(x)

    y = old_model.layers[-4].output
    y = Conv3D(2048, kernel_size=1, strides=1, padding='same', use_bias=False)(y)
    y = BatchNormalization(axis=-1)(y)
    y = Activation('relu')(y)
    y = TimeDistributed(GlobalAveragePooling2D())(y)
    y = Flatten()(y)
    z = Concatenate()([x,y])
    out = Dense(nb_classes, activation='softmax')(z)
    return Model(inputs=old_model.input, outputs=out)

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

    steps_per_epoch = len(train_data) // batch_size
    val_steps_per_epoch = len(test_data) // batch_size

    new_model = rebuild_model(model_path, nb_classes=len(classes))
    new_model.summary(line_length=150)
    sgd = SGD(lr=0.001, momentum=0.99, nesterov=True)
    new_model.compile(optimizer=sgd, loss='categorical_crossentropy',
                      metrics=['accuracy', 'top_k_categorical_accuracy'])

    model_name = 'ShuffleNet_4_12_4_epoch153_finetuning.hdf5'
    checkpointer = ModelCheckpoint(filepath=model_name, verbose=2, save_best_only=True, monitor='val_acc', mode='max')
    reduce_lr = ReduceLROnPlateau(factor=0.1, cooldown=0, patience=10, min_lr=1e-8, verbose=2, monitor='val_loss',
                                  mode='min')
    early_stop = EarlyStopping(monitor='val_loss', patience=40, verbose=2, mode='min')
    callback = [checkpointer, reduce_lr, early_stop]

    new_model.fit_generator(
        generator=frame_generator(batch_size, train_data, frames_per_clip, crop_size, classes, is_train=True),
        steps_per_epoch=steps_per_epoch, epochs=nb_epoch,
        verbose=1, callbacks=callback,
        validation_data=frame_generator(batch_size, test_data, frames_per_clip, crop_size, classes, is_train=False),
        validation_steps=val_steps_per_epoch, use_multiprocessing=True, workers=12)



def main():
    frame_per_clip = 16
    batch_size = 16
    nb_epoch = 200
    image_shape = (224, 224, 3)
    model_path = "ShuffleNet2p1D_4_12_4_epoch153.hdf5"
    print("Model Path: ", model_path)
    train(frame_per_clip, model_path, image_shape=image_shape, batch_size=batch_size, nb_epoch=nb_epoch)


if __name__ == '__main__':
    main()