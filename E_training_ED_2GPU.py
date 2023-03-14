import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from keras.optimizers import SGD, RMSprop, Adam, Nadam
from keras.models import load_model
import keras.backend as K
from C_data_generator import *
from D_Encoder_Decoder import encoder_decoder_model, load_weight

from callbacks.epochcheckpoint import EpochCheckpoint
from callbacks.trainingmonitor import TrainingMonitor
import os
from keras.utils.multi_gpu_utils import multi_gpu_model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


def train(frames_per_clip, image_shape=None, batch_size=32, nb_epoch=100, model_name="ShuffleNet2p1D",
          begin_training=True, start_epoch=0, num_GPU=2):
    crop_size = image_shape[0]

    train_data = get_data('train.csv')
    test_data = get_data('test.csv')

    classes = get_classes(train_data)
    print('Number of classes:', len(classes))
    print('Train set:', len(train_data))
    print('Test set:', len(test_data))

    train_data = clean_data(train_data, frames_per_clip, classes, 3000)
    test_data = clean_data(test_data, frames_per_clip, classes, 3000)

    print('Train set after clean:', len(train_data))
    print('Test set after clean:', len(test_data))

    # total_data = train_data + test_data
    # random.shuffle(test_data)
    # test_data = test_data[:1000]
    # print('Total set after clean:', len(total_data))
    # print('Test set after clean:', len(test_data))

    if not os.path.exists(model_name):
        os.mkdir(model_name)
    if not os.path.exists(os.path.join(model_name, "checkpoints")):
        os.mkdir(os.path.join(model_name, "checkpoints"))

    steps_per_epoch = len(train_data) // batch_size
    val_steps_per_epoch = len(test_data) // batch_size


    filepath = os.path.join(model_name, model_name + ".hdf5")
    checkpoint_path = os.path.join(model_name, 'checkpoints')


    adam = Adam(lr=0.0002, beta_1=0.5)
    early_stopper = EarlyStopping(patience=200, mode='min', verbose=2, monitor='val_loss')
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=2, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr = ReduceLROnPlateau(factor=0.1, cooldown=0, patience=5, min_lr=1e-8, verbose=2, monitor='val_loss',
                                  mode='min')
    log_path = os.path.join(model_name, "logs")
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    tensorboard = TensorBoard(log_dir=log_path)

    if num_GPU > 1:
        epoch_checkpoint = EpochCheckpoint(checkpoint_path, every=1, startAt=start_epoch, multi_GPU=True)
    else:
        epoch_checkpoint = EpochCheckpoint(checkpoint_path, every=1, startAt=start_epoch, multi_GPU=False)

    callback = [early_stopper, checkpointer, reduce_lr, tensorboard, epoch_checkpoint]

    with tf.device("/cpu:0"):
        if begin_training:
            model = encoder_decoder_model(input_shape=(frames_per_clip, crop_size, crop_size, 3))
            # effcient_model_path = 'EfficientNet3D/EfficientNet3D_.hdf5'
            # model = load_weight(model, effcient_model_path)
            model.summary(line_length=150)
        else:
            modelpath = os.path.join(model_name, 'checkpoints', 'epoch_' + str(start_epoch) + '.hdf5')
            print("LOAD MODEL:", modelpath)
            model = load_model(modelpath)
            model.summary(line_length=150)

    multi_GPU_model = multi_gpu_model(model, gpus=num_GPU)
    multi_GPU_model.compile(optimizer=adam, loss='mse')
    hist = multi_GPU_model.fit_generator(
        generator=generator_encoder_decoder(batch_size, train_data, frames_per_clip, crop_size, is_train=True),
        steps_per_epoch=steps_per_epoch, epochs=nb_epoch,
        verbose=1, callbacks=callback,
        validation_data=generator_encoder_decoder(batch_size, test_data, frames_per_clip, crop_size, is_train=False),
        validation_steps=val_steps_per_epoch, use_multiprocessing=True, workers=12)


def main():
    frame_per_clip = 16
    nb_epoch = 10000
    image_shape = (224, 224, 3)
    num_GPU = 2

    model_name = "Encode_Decoder"
    BEGIN_TRAINING = False
    if BEGIN_TRAINING:
        START_EPOCH = 0
    else:
        START_EPOCH = 9
    batch_size = 6 * num_GPU


    train(frame_per_clip, image_shape=image_shape, batch_size=batch_size, nb_epoch=nb_epoch,
          model_name=model_name, begin_training=BEGIN_TRAINING, start_epoch=START_EPOCH, num_GPU=num_GPU)

if __name__ == '__main__':
    main()
