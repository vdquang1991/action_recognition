import csv
import os
import glob
import random
import numpy as np
import cv2
from PIL import Image
from keras.utils import to_categorical
from keras.applications.resnet50 import preprocess_input


def get_data(csv_file):
    """Load our data from file."""
    with open(csv_file, 'r') as fin:
        reader = csv.reader(fin)
        data = list(reader)
    return data

def split_train_test(data):
    train, test = [], []
    for item in data:
        if item[0] == 'train' or item[0] == 'TRAIN':
            train.append(item)
        else:
            test.append(item)
    return train, test

def get_classes(data):
    """Extract the classes from our data. If we want to limit them,
    only return the classes we need."""
    classes = []
    for item in data:
        if item[1] not in classes:
            classes.append(item[1])
    # Sort them.
    classes = sorted(classes)
    return classes

def clean_data(data, CLIPS_LENGTH, classes, MAX_FRAMES=3000):
    """Limit samples to greater than the sequence length and fewer
    than N frames. Also limit it to classes we want to use."""
    data_clean = []
    for item in data:
        if int(item[3]) >= CLIPS_LENGTH and int(item[3]) <= MAX_FRAMES and item[1] in classes:
            data_clean.append(item)
    return data_clean

def get_class_one_hot(class_str, classes):
    label_encoded = classes.index(class_str)
    # Now one-hot it.
    label_hot = to_categorical(label_encoded, len(classes))
    assert len(label_hot) == len(classes)
    return label_hot

def get_frames_for_sample(sample):
    """Given a sample row from the data file, get all the corresponding frame
    filenames."""
    path = os.path.join(sample[0], sample[1])
    folder_name = sample[2]
    images = sorted(glob.glob(os.path.join(path, folder_name + '/*jpg')))
    num_frames = sample[3]
    return images, int(num_frames)


def read_images(frames, start_idx, num_frames_per_clip):
    img_data = []
    for i in range(start_idx, start_idx + num_frames_per_clip):
        img = Image.open(frames[i])
        img = np.asarray(img)
        img_data.append(img)
    return img_data

def read_images_1(frames, num_frames, num_frames_per_clip):
    img_data = []
    for i in range(0, num_frames):
        img = Image.open(frames[i])
        img = np.asarray(img)
        img_data.append(img)
    for i in range(num_frames_per_clip-num_frames):
        img = Image.open(frames[num_frames-1])
        img = np.asarray(img)
        img_data.append(img)
    return img_data

def data_process(tmp_data, crop_size, is_train):
    img_datas = []
    mean = np.asarray([0.485, 0.456, 0.406])
    std = np.asarray([0.229, 0.224, 0.225])
    crop_x = 0
    crop_y = 0
    if crop_size==224:
        resize_value=256
    else:
        resize_value=129

    if is_train == True and random.random() > 0.5:
        cvt_color = True
    else:
        cvt_color = False

    if is_train == True and random.random() > 0.5:
        flip = True
    else:
        flip = False

    for j in range(len(tmp_data)):
        img = Image.fromarray(tmp_data[j].astype(np.uint8))
        if img.width > img.height:
            scale = float(resize_value) / float(img.height)
            img = np.array(cv2.resize(np.array(img), (int(img.width * scale + 1), resize_value))).astype(np.float32)
        else:
            scale = float(resize_value) / float(img.width)
            img = np.array(cv2.resize(np.array(img), (resize_value, int(img.height * scale + 1)))).astype(np.float32)
        if j == 0:
            if is_train:
                crop_x = random.randint(0, int(img.shape[0] - crop_size))
                crop_y = random.randint(0, int(img.shape[1] - crop_size))
            else:
                crop_x = int((img.shape[0] - crop_size) / 2)
                crop_y = int((img.shape[1] - crop_size) / 2)
        img = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :]
        img = np.asarray(img) / 255.0

        if cvt_color:
           img = -img
        if flip:
            img = np.flip(img, axis=1)

        img_datas.append(img)
    return img_datas

def frame_generator(batch_size, data, num_frames_per_clip, crop_size, classes, is_train=True):
    while True:
        X, y = [], []
        random.shuffle(data)
        for i in range(len(data)):
            row = data[i]
            frames, num_frames = get_frames_for_sample(row)
            if num_frames >= num_frames_per_clip:
                start_idx = random.randint(0, num_frames - num_frames_per_clip)
                rgb_img_data = read_images(frames, start_idx, num_frames_per_clip)
            else:
                rgb_img_data = read_images_1(frames, num_frames, num_frames_per_clip)
            rgb_img_data = data_process(rgb_img_data, crop_size, is_train)
            rgb_img_data = np.asarray(rgb_img_data)
            X.append(rgb_img_data)
            label = get_class_one_hot(row[1], classes=classes)
            y.append(label)
            if len(y)== batch_size:
                yield np.asarray(X), np.asarray(y)
                X, y =[], []

#----------------------------------------------------------------------------------------------------------------------------
#------------------------------------------GENERATOR FOR IMAGE---------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------

def img_process(rgb_img, crop_size, is_train=True):
    if is_train == True and random.random() > 0.5:
        cvt_color = True
    else:
        cvt_color = False

    if is_train == True and random.random() > 0.5:
        flip = True
    else:
        flip = False

    img = Image.fromarray(rgb_img.astype(np.uint8))
    if img.width > img.height:
        scale = float(256) / float(img.height)
        img = np.array(cv2.resize(np.array(img), (int(img.width * scale + 1), 256))).astype(np.float32)
    else:
        scale = float(256) / float(img.width)
        img = np.array(cv2.resize(np.array(img), (256, int(img.height * scale + 1)))).astype(np.float32)

    if is_train:
        crop_x = random.randint(0, int(img.shape[0] - crop_size))
        crop_y = random.randint(0, int(img.shape[1] - crop_size))
    else:
        crop_x = int((img.shape[0] - crop_size) / 2)
        crop_y = int((img.shape[1] - crop_size) / 2)

    img = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :]
    img = np.asarray(img) / 255.

    if cvt_color:
        img = -img
    if flip:
        img = np.flip(img, axis=1)
    return np.asarray(img)

def img_generator(batch_size, data, crop_size, classes, is_train=True):
    while True:
        X, y = [], []
        random.shuffle(data)
        for i in range(len(data)):
            row = data[i]
            frames, num_frames = get_frames_for_sample(row)
            idx = random.randint(0, num_frames-1)
            rgb_img = Image.open(frames[idx])
            # rgb_img = rgb_img.resize((crop_size,crop_size))
            # rgb_img = np.asarray(rgb_img)
            # rgb_img = preprocess_input(rgb_img)
            rgb_img = np.asarray(rgb_img)
            rgb_img = img_process(rgb_img, crop_size,is_train)
            X.append(rgb_img)
            label = get_class_one_hot(row[1], classes=classes)
            y.append(label)
            if len(y)== batch_size:
                yield np.asarray(X), np.asarray(y)
                X, y =[], []

#----------------------------------------------------------------------------------------------------------------------------
#------------------------------------------GENERATOR FOR TEACHER STUDENT---------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------

def generator_teacher_student(batch_size, data, num_frames_per_clip, crop_size, is_train=True):
    while True:
        X_img, X_frames, y = [], [], []
        random.shuffle(data)
        for i in range(len(data)):
            row = data[i]
            frames, num_frames = get_frames_for_sample(row)
            if num_frames >= num_frames_per_clip:
                start_idx = random.randint(0, num_frames - num_frames_per_clip)
                rgb_img_data = read_images(frames, start_idx, num_frames_per_clip)
            else:
                rgb_img_data = read_images_1(frames, num_frames, num_frames_per_clip)
            rgb_img_data = data_process(rgb_img_data, crop_size, is_train)
            idx = random.randint(0, len(rgb_img_data)-1)
            rgb_img = rgb_img_data[idx]
            rgb_img = np.asarray(rgb_img)
            rgb_img_data = np.asarray(rgb_img_data)
            X_img.append(rgb_img)
            X_frames.append(rgb_img_data)

            label = 1
            y.append(label)
            if len(y) == batch_size:
                yield [np.asarray(X_img), np.asarray(X_frames), np.asarray(X_frames)], np.asarray(y)
                X_img, X_frames, y = [], [], []


#----------------------------------------------------------------------------------------------------------------------------
#------------------------------------------GENERATOR FOR ENCODER PREDICT-----------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------
def generator_encoder_predict(batch_size, data, num_frames_per_clip, crop_size, classes, is_train=True):
    while True:
        X, y = [], []
        random.shuffle(data)
        for i in range(len(data)):
            row = data[i]
            frames, num_frames = get_frames_for_sample(row)
            if num_frames > num_frames_per_clip:
                start_idx = random.randint(0, num_frames - num_frames_per_clip)
                end_idx = start_idx + num_frames_per_clip - 1
            else:
                start_idx = 0
                end_idx = num_frames - 1
            rgb_start = Image.open(frames[start_idx])
            rgb_start = np.asarray(rgb_start)
            rgb_end = Image.open(frames[end_idx])
            rgb_end = np.asarray(rgb_end)
            rgb_start = img_process(rgb_start, crop_size, is_train)
            rgb_end = img_process(rgb_end, crop_size, is_train)

            temp = []
            for k in range(num_frames_per_clip//2):
                temp.append(rgb_start)
            for k in range(num_frames_per_clip//2):
                temp.append(rgb_end)
            temp = np.asarray(temp)
            X.append(temp)
            label = get_class_one_hot(row[1], classes=classes)
            y.append(label)
            if len(y) == batch_size:
                yield np.asarray(X), np.asarray(y)
                X, y = [], []


#----------------------------------------------------------------------------------------------------------------------------
#------------------------------------------GENERATOR FOR ENCODER DECODER-----------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------
def generator_encoder_decoder(batch_size, data, num_frames_per_clip, crop_size, is_train=True):
    while True:
        X, y = [], []
        random.shuffle(data)
        for i in range(len(data)):
            row = data[i]
            frames, num_frames = get_frames_for_sample(row)
            if num_frames > num_frames_per_clip:
                start_idx = random.randint(0, num_frames - num_frames_per_clip)
                rgb_img_data = read_images(frames, start_idx, num_frames_per_clip)
            else:
                rgb_img_data = read_images_1(frames, num_frames, num_frames_per_clip)
            rgb_img_data = data_process(rgb_img_data, crop_size, is_train)

            temp = []
            for j in range(num_frames_per_clip//2):
                temp.append(rgb_img_data[0])
            for j in range(num_frames_per_clip//2):
                temp.append(rgb_img_data[-1])
            rgb_img_data = np.asarray(rgb_img_data)
            temp = np.asarray(temp)
            y.append(rgb_img_data)
            X.append(temp)
            if len(y)== batch_size:
                yield np.asarray(X), np.asarray(y)
                X, y =[], []


