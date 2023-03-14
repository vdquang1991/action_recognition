from keras.models import load_model
from D_ShuffleNet2p1D import ShuffleNet_2p1D
from D_Encoder_Decoder import EfficientB3Net
from keras import Model
from keras.layers import Activation

# model = load_model('saved_models/Encode_Decoder/Encode_Decoder.hdf5')
model = load_model('saved_models/Predict_Noise_ResNet2p1D_18/Predict_Noise_ResNet2p1D_18_2_2_2_2_.hdf5')
# model = load_model('saved_models/Encode_Decoder/checkpoints/epoch_6.hdf5')
# model = ShuffleNet_2p1D(input_shape=(16, 224, 224, 3), nb_classes=400, pooling='avg', num_shuffle_units=[4, 12, 4])
# model = EfficientB3Net((16,224,224,3),400)
# model.load_weights('saved_models/epoch_70.h5')
# model = model.layers[-2]
model.summary(line_length=150)
print(len(model.layers))
# model.save('Encode_Decoder.hdf5')

# print(len(model.layers))
# #
# # x = model.layers[-4].output
# y = model.layers[376].output
# y = Activation('relu')(y)
# new_model = Model(model.input, [y])
# new_model.summary(line_length=150)

