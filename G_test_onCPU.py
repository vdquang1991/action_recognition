import cv2
import time
import numpy as np
from keras import Model
from keras.models import load_model
from keras.layers import Flatten, Dense, BatchNormalization, Activation, TimeDistributed, GlobalAveragePooling2D
from keras.layers import GlobalAveragePooling3D, Concatenate


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


# model_path = 'saved_models/Distillation_Model/distilled_shuffleNet.hdf5'
# new_model = rebuild_model(model_path, nb_classes=51)
# new_model.summary(line_length=150)

# filepath = 'saved_models/other_models/ResNet2p1D_18.hdf5'
filepath = 'saved_models/Distillation_Model/distilled_shuffleNet.hdf5'
new_model = load_model(filepath)

cap = cv2.VideoCapture('2.mp4')

# Read until video is completed
start = time.time()
num_frames = 0
clip = []
ave_running_time = []
class_string = 'Classes = -1'

while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    frame = cv2.resize(frame, (224,224))
    img = np.asarray(frame) / 127.5
    img -= img
    num_frames +=1
    clip.append(img)
    if len(clip)==16:
        X_predict = np.asarray(clip)
        X_predict = np.reshape(X_predict, newshape=(1,16,224,224,3))
        # print(X_predict.shape)
        start_running_time = time.time()
        score = new_model.predict(X_predict)
        end_running_time = time.time()
        class_predict = np.argmax(score)
        ave_running_time.append(end_running_time-start_running_time)
        class_string = 'Classes = ' + str(class_predict)
        clip = []

    frame = cv2.resize(frame, (600, 600))
    cv2.putText(frame, class_string, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  else:
    break
end = time.time()

seconds = end - start
print("Time taken : {0} seconds".format(seconds))
fps  = num_frames / seconds
print("Estimated frames per second : {0}".format(fps))

ave_running_time = np.asarray(ave_running_time)
mean_time = np.mean(ave_running_time)
print('Ave running time = ', mean_time)
print('FPS = ', 16/mean_time)


# When everything done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()
