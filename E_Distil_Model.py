import os
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

from keras.models import load_model
from keras import Model
from keras.applications.resnet50 import ResNet50
from keras.applications import ResNet101V2, ResNet152V2
from keras.layers import Flatten, GlobalAveragePooling2D, Dense, BatchNormalization, Activation, Reshape, Lambda
from keras.layers import Input, Dense, Concatenate, Conv3D
from D_ShuffleNet2p1D import ShuffleNet_2p1D
from D_ShuffleNet3D import ShuffleNet_3D
from keras.losses import mean_squared_error
from keras.optimizers import Adam
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau

import keras.backend as K

from C1_data_generator import *
from callbacks.epochcheckpoint import EpochCheckpoint
from callbacks.trainingmonitor import TrainingMonitor

from keras.utils.multi_gpu_utils import multi_gpu_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

def loss_func(input_distillation, alpha=0.9, beta=0.1):
    out7x7, out14x14, out7x7_student, out14x14_student = input_distillation
    out7x7 = Flatten()(out7x7)
    out14x14 = Flatten()(out14x14)
    out7x7_student = Flatten()(out7x7_student)
    out14x14_student = Flatten()(out14x14_student)

    loss_14x14 = mean_squared_error(out14x14, out14x14_student)
    loss_7x7 = mean_squared_error(out7x7, out7x7_student)
    loss = alpha*loss_7x7 + beta*loss_14x14
    return loss

# def convert_gpu_model(org_model: Model) -> Model:
#     train_model = org_model
#     return train_model


class TrainingCallback(Callback):
    def __init__(self, model, model_path):
        super(TrainingCallback, self).__init__()
        self.model = model
        self.model_path = model_path

    def on_epoch_end(self, epoch, logs=None):
        save_model_path = os.path.join(self.model_path, "distilled_shuffleNet.hdf5")
        model.born_again_model.save(save_model_path)


class BornAgainModel(object):
    def __init__(self, input_shape_teacher_1, input_shape_student, num_classes, multi_GPU=False, num_GPU=1):
        self.train_model, self.born_again_model = None, None
        self.multi_GPU = multi_GPU
        self.num_GPU = num_GPU
        if self.multi_GPU:
            with tf.device("/cpu:0"):
                self.train_model, self.born_again_model = self.prepare(input_shape_teacher_1, input_shape_student, num_classes)
        else:
            self.train_model, self.born_again_model = self.prepare(input_shape_teacher_1, input_shape_student, num_classes)
        if self.multi_GPU:
            self.multi_GPU_model = multi_gpu_model(self.train_model, gpus=self.num_GPU)
        # self.train_model = convert_gpu_model(self.train_model)

    def prepare(self, input_shape_teacher_1, input_shape_student, num_classes):
        inp_teacher1 = Input(shape=input_shape_teacher_1, name='input_teacher_1')
        teacher1_model = ResNet101V2(include_top=False, weights='imagenet', input_tensor=inp_teacher1, input_shape=None,pooling=None)
        for layer in teacher1_model.layers:
            layer.trainable = False
        out7x7_teacher1 = teacher1_model.output
        out7x7_teacher1 = Reshape(target_shape=(1,7,7,2048))(out7x7_teacher1)
        out14x14_teacher1 = teacher1_model.layers[330].output
        out14x14_teacher1 = Reshape(target_shape=(1,14,14,1024))(out14x14_teacher1)
        out14x14_teacher1 = Concatenate(axis=1)([out14x14_teacher1, out14x14_teacher1])

        teacher2_model = load_model("ResNet3D_50_16.hdf5")
        for layer in teacher2_model.layers:
            layer.trainable = False
        out7x7_teacher2 = teacher2_model.layers[-3].output
        out14x14_teacher2 = teacher2_model.layers[137].output

        out7x7 = Concatenate(axis=1)([out7x7_teacher1, out7x7_teacher2])
        out14x14 = Concatenate(axis=1)([out14x14_teacher1, out14x14_teacher2])


        student_model = ShuffleNet_3D(input_shape=input_shape_student, nb_classes=num_classes, pooling='avg', num_shuffle_units=[3, 3, 3])
        student_model.load_weights("Shuffle2p1D_16_14_16.hdf5")

        out7x7_student = student_model.layers[-4].output
        out7x7_student = Conv3D(2048, kernel_size=(1,1,1), strides=(1,1,1), padding='same', use_bias='False', name='conv3d_student_7x7')(out7x7_student)
        out7x7_student = BatchNormalization(name='bn_student_7x7')(out7x7_student)
        out7x7_student = Activation('relu', name='relu_student_7x7')(out7x7_student)

        out14x14_student = student_model.layers[129].output
        out14x14_student = Conv3D(1024, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same', use_bias='False',
                                name='conv3d_student_14x14')(out14x14_student)
        out14x14_student = BatchNormalization(name='bn_student_14x14')(out14x14_student)
        out14x14_student = Activation('relu', name='relu_student_14x14')(out14x14_student)

        inps = [inp_teacher1, teacher2_model.input, student_model.input]

        output_loss = Lambda(loss_func, output_shape=(1,), name='kd_')(
            [out7x7, out14x14, out7x7_student, out14x14_student]
        )
        born_again_model = Model(inputs=student_model.input, outputs=[out7x7_student, out14x14_student])
        train_model = Model(inputs=inps, outputs=output_loss)

        return train_model, born_again_model


frames_per_clip = 16
nb_epoch = 10000
image_shape = (224,224,3)
begin_training = True
use_multi_GPU = False
num_GPU = 2

if begin_training:
    start_epoch = 0
else:
    start_epoch=15

if use_multi_GPU:
    batch_size = 8 * num_GPU
else:
    batch_size = 8

crop_size = image_shape[0]

train_data = get_data('train.csv')
test_data = get_data('test.csv')

classes = get_classes(train_data)
print('Number of classes:', len(classes))
print('Train set:', len(train_data))
print('Test set:', len(test_data))

train_data = clean_data(train_data, 8, classes, 3000)
test_data = clean_data(test_data, 8, classes, 3000)
print('Train set after clean:', len(train_data))
print('Test set after clean:', len(test_data))

total_data = train_data + test_data
print('Total set after clean:', len(total_data))

steps_per_epoch = len(total_data) // batch_size


model = BornAgainModel(input_shape_teacher_1=(crop_size,crop_size,3), input_shape_student=(frames_per_clip,crop_size,crop_size,3),
                       num_classes=len(classes), multi_GPU=use_multi_GPU, num_GPU=num_GPU)
print("MODEL STUDENT")
model.born_again_model.summary(line_length=150)
print("TRAINING MODEL")
model.train_model.summary(line_length=150)


model_path = "Distillation_Model"
if not os.path.exists(model_path):
    os.mkdir(model_path)
if not os.path.exists(os.path.join(model_path, "output")):
    os.mkdir(os.path.join(model_path, "output"))
if not os.path.exists(os.path.join(model_path, "checkpoints")):
    os.mkdir(os.path.join(model_path, "checkpoints"))

filepath = os.path.join(model_path, 'best_model.hdf5')
checkpoint_path = os.path.join(model_path, 'checkpoints')

training_callback = TrainingCallback(model, model_path)
checkpoint = ModelCheckpoint(filepath=filepath, monitor='loss', verbose=2, save_best_only=True)
lr_reducer = ReduceLROnPlateau(factor=0.1, cooldown=0, patience=10, min_lr=0.5e-7, verbose=2,
                               monitor='loss', mode='min')

callbacks = [checkpoint, lr_reducer, training_callback,
             EpochCheckpoint(checkpoint_path, every=5, startAt=start_epoch)]

if begin_training:
    if use_multi_GPU == False:
        model.train_model.compile(
            optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
            loss=lambda y_true, y_pred: y_pred,
        )
    else:
        model.multi_GPU_model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
            loss=lambda y_true, y_pred: y_pred,)
else:
    modelpath =  os.path.join(model_path, 'checkpoints', 'epoch_' + str(start_epoch) + '.hdf5')
    print("LOAD MODEL")
    # model.train_model = load_model(modelpath)
    model.train_model.load_weights(modelpath)
    model.train_model.summary(line_length=150)

if use_multi_GPU == False:
    model.train_model.fit_generator(generator=generator_teacher_student(batch_size, train_data, frames_per_clip, crop_size, is_train=True),
                        steps_per_epoch=steps_per_epoch, epochs=nb_epoch,
                        verbose=1, callbacks=callbacks, use_multiprocessing=True, workers=12)
else:
    model.multi_GPU_model.fit_generator(generator=generator_teacher_student(batch_size, train_data, frames_per_clip, crop_size, is_train=True),
                        steps_per_epoch=steps_per_epoch, epochs=nb_epoch,
                        verbose=1, callbacks=callbacks, use_multiprocessing=True, workers=12)


