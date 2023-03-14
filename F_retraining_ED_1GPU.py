import tensorflow as tf
from keras import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from keras.optimizers import SGD, RMSprop, Adam, Nadam
from keras.layers import GlobalAveragePooling2D, TimeDistributed, Dense, Flatten, Activation
from keras.models import load_model
import keras.backend as K
from C_data_generator import *

from callbacks.epochcheckpoint import EpochCheckpoint
from callbacks.trainingmonitor import TrainingMonitor
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

def swish(x):
    return x * K.sigmoid(x)

def rebuild_model(encoder_decoder_model_path, nb_classes):
    old_model = load_model(encoder_decoder_model_path)
    x = old_model.layers[376].output
    # x = Activation(swish)(x)
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    x = Flatten()(x)
    out = Dense(nb_classes, activation='softmax')(x)
    model = Model(inputs=old_model.input, outputs=out)
    return model


def fine_tune(frames_per_clip, image_shape=None, batch_size=32, nb_epoch=100, model_name="ShuffleNet2p1D",
          begin_training=True, start_epoch=0, num_GPU=1, en_de_model_path=''):
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
    # print('Total set after clean:', len(total_data))

    if not os.path.exists(model_name):
        os.mkdir(model_name)
    if not os.path.exists(os.path.join(model_name, "output")):
        os.mkdir(os.path.join(model_name, "output"))
    if not os.path.exists(os.path.join(model_name, "checkpoints")):
        os.mkdir(os.path.join(model_name, "checkpoints"))

    steps_per_epoch = len(train_data) // batch_size
    val_steps_per_epoch = len(test_data) // batch_size
    checkpoint_path = os.path.join(model_name, 'checkpoints')

    filepath = os.path.join(model_name, model_name + ".hdf5")
    plotPath = os.path.join(model_name, "output")
    jsonPath = os.path.join(model_name, "output")
    jsonName = model_name + '_result.json'

    sgd = SGD(lr=0.001, momentum=0.99, nesterov=True)
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
        model = rebuild_model(en_de_model_path, len(classes))
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
        generator=frame_generator(batch_size, train_data, frames_per_clip, crop_size, classes, is_train=True),
        steps_per_epoch=steps_per_epoch, epochs=nb_epoch,
        verbose=1, callbacks=callback,
        validation_data=frame_generator(batch_size, test_data, frames_per_clip, crop_size, classes, is_train=False),
        validation_steps=val_steps_per_epoch, use_multiprocessing=True, workers=12)


def main():
    frame_per_clip = 16
    nb_epoch = 200
    image_shape = (224, 224, 3)
    num_GPU = 1

    model_name = "Encoder_Decoder_Fine_Tune"
    en_de_model_path = 'Encode_Decoder.hdf5'
    BEGIN_TRAINING = True
    if BEGIN_TRAINING:
        START_EPOCH = 0
    else:
        START_EPOCH = 6
    batch_size = 16 * num_GPU

    print('************************************************************************')
    print('Model name: ', model_name)
    print('Number of GPUs: ', num_GPU)
    print('Number of frames: ', frame_per_clip)
    print('Image shape: ', image_shape)
    print('Begin training: ', BEGIN_TRAINING)
    print('Start epoch: ', START_EPOCH)
    print('Batch size=', batch_size)
    print('************************************************************************')

    fine_tune(frame_per_clip, image_shape=image_shape, batch_size=batch_size, nb_epoch=nb_epoch, model_name=model_name,
              begin_training=BEGIN_TRAINING, start_epoch=START_EPOCH, num_GPU=num_GPU, en_de_model_path=en_de_model_path)

if __name__ == '__main__':
    main()
