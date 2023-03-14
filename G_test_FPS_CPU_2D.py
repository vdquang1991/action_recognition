import cv2
import time
import numpy as np
from keras.applications import ResNet50

cap = cv2.VideoCapture('2.mp4')
img_shape = (224, 224, 3)
new_model = ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=img_shape, pooling='ave')
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

    X_predict = np.asarray(img)
    X_predict = np.reshape(X_predict, newshape=(1,224,224,3))

    start_running_time = time.time()
    score = new_model.predict(X_predict)
    end_running_time = time.time()

    ave_running_time.append(end_running_time-start_running_time)


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
print('FPS = ', 1/mean_time)


# When everything done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()
