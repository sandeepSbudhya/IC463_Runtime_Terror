from __future__ import division
from tkinter import *
import math
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import requests
import cv2
from util import *
import argparse
import os
import os.path as osp
from DNModel import net
from img_process import preprocess_img, inp_to_image
import pandas as pd
import random
import pickle as pkl
import tensorflow as tf
from keras import Input, layers
from keras import optimizers
from keras.preprocessing import image
from keras.models import Model
from keras.applications.densenet import DenseNet201
from keras.preprocessing import image
from keras.applications.densenet import preprocess_input


import pyrebase
import time
import firebase_admin
from firebase_admin import credentials, firestore, storage


# Disable all GPUS for tensorflow, comment if VRAM > 4GB
try:

    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


model_prediction = tf.keras.models.load_model('model')
model_dense = DenseNet201(weights='imagenet')
# model_dense = keras.applications.DenseNet201(weights='imagenet')

model_new = Model(model_dense.input, model_dense.layers[-2].output)


config = {
    "apiKey": "AIzaSyBVsJbZO4MaT4wrsmUw98t_HCyZwEE-EsQ",
    "authDomain": "accident-detection-54513.firebaseapp.com",
    "databaseURL": "https://accident-detection-54513.firebaseio.com/",
    "projectId": "accident-detection-54513",
    "storageBucket": "accident-detection-54513.appspot.com",
    "messagingSenderId": "115519272790",
    "appId": "1:115519272790:web:35ad75e3851d9492be48f7"
}

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
# load the firebase crediantials from a json file
cred = credentials.Certificate('firebase_credentials.json')
firebase_admin.initialize_app(cred)


def send_to_db(image_path, Longitude=77, Latitude=13):

    name = time.time()
    name *= 1000
    name = int(name)
    storage.child(str(name)).put(image_path)

    url = storage.child(str(name)).get_url(2)

    db = firestore.client()

    ref = db.collection('accidents')

    ref.add({
        "URL": url,
        "Longitude": Longitude,
        "Latitude": Latitude,
        "is_dismissed": False,
        "is_reported": False,
        "timestamp": name

    })


def extract_features(image_path):
    # to prevent multiple frames of the same accident from being sent to db
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    global last_accident

    x = np.expand_dims(x, axis=0)
    x = np.array(x, dtype='float64')
    x = preprocess_input(x)

    features = model_new.predict(x)
    features = features.flatten()

    S = [features]
    S = np.asarray(S)
    x = model_prediction.predict(S)

    print(x)

    if (x >= 0.2):

        current_acc = time.time()

        global last_accident
        if (current_acc) - (last_accident) < 1.2:  # Code to check for time delay b/w accidents
            print('skipped')
        else:
            image.save_img("accident_pics/accident{}.jpg".format(
                str(current_acc) + "_"+str(x)), img)

            path_to_img = "accident_pics/accident{}.jpg".format(
                str(current_acc) + "_"+str(x))
            send_to_db(image_path=path_to_img)
            print('save img and put to db')
        last_accident = current_acc


# Checking for collision between objects from bounding box
def checkCollisions(value):
    print('checking')
    for i in range(len(x_y_values)):
        for j in range(i+1, len(x_y_values)):
            box1 = x_y_values[i]
            box2 = x_y_values[j]
            if proximityCheck(box1, box2):
                crop(box1, box2, value)


# Checking for proximity between ojects from bounding box
def proximityCheck(b1, b2):
    print('prox')

    b1x = b1[4][0]+(b1[4][1]/2)
    b2x = b2[4][0]+(b2[4][1]/2)

    b1y = b1[5][0]+(b1[5][1]/2)
    b2y = b2[5][0]+(b2[5][1]/2)

    halfWidthSumX = (b1[4][1]/2)+(b2[4][1])/2
    halfWidthSumY = (b1[5][1]/2)+(b2[5][1])/2

    distBetweenCentersX = abs(b1x-b2x)
    distBetweenCentersY = abs(b1y-b2y)

    differenceX = distBetweenCentersX-halfWidthSumX
    differenceY = distBetweenCentersY-halfWidthSumY

    print(differenceX, differenceY)
    # If boxes completely overlap
    if differenceX < 0 and differenceY < 0:
        print('collision')
        return True
    # If Difference is less than 5% of the sum of the width of both boxes
    if differenceX < (0.05*halfWidthSumX) and differenceY < (0.05*halfWidthSumY):
        print('coming close')
        return True
    else:
        return False


def crop(b1, b2, value):
    minx = min(b1[0][0], b2[0][0])
    miny = min(b1[0][1], b2[0][1])
    maxx = max(b1[2][0], b2[2][0])
    maxy = max(b1[2][1], b2[2][1])
    img = cv2.imread(value)
    h, w = img.shape[:2]

    crop_img = img[max(0, int(miny)-50):min(h, int(maxy)+50),
                   max(0, int(minx)-50):min(w, int(maxx)+50)]
    cv2.imwrite("cropped.jpg", crop_img)
    extract_features('cropped.jpg')


# global variables
scales = "1,2,3"
batch_size = 1
confidence = 0.7
nms_thesh = 0.5
start = 0
url = 'http://192.168.0.146:8080/shot.jpg'

CUDA = torch.cuda.is_available()
print(CUDA)

num_classes = 80
classes = load_classes('data/coco.names')

model = net('cfg/yolov3.cfg')
model.load_weights('cfg/yolov3.weights')
print("Networkloaded")

model.DNInfo["height"] = '256'
in_dim = int(model.DNInfo["height"])

x_y_values = []
temp = []
cc = 0
last_accident = 0

# Object dettection code


def object_detection(image_path):
    global x_y_values
    x_y_values = []
    images = image_path

    if CUDA:
        model.cuda()
    model.eval()

    read_dir = time.time()
    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images) if os.path.splitext(
            img)[1] == '.png' or os.path.splitext(img)[1] == '.jpeg' or os.path.splitext(img)[1] == '.jpg']
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print("No with the name {}".format(images))
        exit()

    if not os.path.exists('result'):
        os.makedirs('result')

    batches = list(map(preprocess_img, imlist, [
                   in_dim for x in range(len(imlist))]))
    im_batches = [x[0] for x in batches]
    orig_ims = [x[1] for x in batches]
    im_dim_list = [x[2] for x in batches]
    # Explain
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

    if CUDA:
        im_dim_list = im_dim_list.cuda()

    leftover = 0

    if (len(im_dim_list) % batch_size):
        leftover = 1

    i = 0

    write = False

    objs = {}

    for batch in im_batches:
        if CUDA:
            batch = batch.cuda()
        #print('batch size => ', batch.size())
        with torch.no_grad():
            prediction = model(batch, CUDA)

        prediction = write_results(
            prediction, confidence, num_classes, nms=True, nms_conf=nms_thesh)

        if type(prediction) == int:
            i += 1
            continue

        # Add the current batch number
        prediction[:, 0] += i*batch_size

        if not write:
            output = prediction
            write = 1
        else:
            output = torch.cat((output, prediction))

        for im_num, image in enumerate(imlist[i*batch_size: min((i + 1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            print("----------------------------------------------------------")
        i += 1

        if CUDA:
            torch.cuda.synchronize()

    try:
        output
    except NameError:
        print("No detections were made")
        exit()

    im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

    scaling_factor = torch.min(in_dim/im_dim_list, 1)[0].view(-1, 1)

    output[:, [1, 3]] -= (in_dim - scaling_factor *
                          im_dim_list[:, 0].view(-1, 1))/2
    output[:, [2, 4]] -= (in_dim - scaling_factor *
                          im_dim_list[:, 1].view(-1, 1))/2

    output[:, 1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1, 3]] = torch.clamp(
            output[i, [1, 3]], 0.0, im_dim_list[i, 0])
        output[i, [2, 4]] = torch.clamp(
            output[i, [2, 4]], 0.0, im_dim_list[i, 1])

    colors = pkl.load(open("pallete", "rb"))

    def write(x, batches, results):
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        img = results[int(x[0])]
        cls = int(x[-1])
        label = "{0}".format(classes[cls])
        print(label)
        if label == 'car' or label == 'car':
            x1 = int(c1[0])
            y1 = int(c1[1])
            x2 = int(c2[0])
            y2 = int(c2[1])
            global temp
            global x_y_values
            temp = []
            temp.append([x1, y1])
            temp.append([x2, y1])
            temp.append([x2, y2])
            temp.append([x1, y2])
            temp.append([x1, x2-x1])  # For Half Width Calculation
            temp.append([y1, abs(y2-y1)])  # For Half Width Calculation

            x_y_values.append(temp)

        color = random.choice(colors)
        cv2.rectangle(img, c1, c2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4),
                    cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
        global cc
        cv2.imwrite('boxes/pls{}.jpg'.format(cc), img)
        cc += 1
        return img

    list(map(lambda x: write(x, im_batches, orig_ims), output))

    det_names = pd.Series(imlist).apply(
        lambda x: "{}/det_{}".format('result', x.split("\\")[-1]))

    list(map(cv2.imwrite, det_names, orig_ims))

    torch.cuda.empty_cache()


def start_from_here():
    # cap = cv2.VideoCapture(path)
    j = 0
    i = 0
    # prev_frame = 0

    while True:

        img = requests.get(url)
        img_arr = np.array(bytearray(img.content), dtype=np.uint8)
        image = cv2.imdecode(img_arr, -1)

        # cv2.imshow('frame', image)

        if True:
            cv2.imshow('frame', image)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
            frame = cv2.resize(image, (400, 300))  # Frame resize
            # variable i to keep track of the number of frames being sent for evaluation
            if i % 5 == 0:
                j = j+1
                s = 'image.jpg'
                cv2.imwrite(s, image)
                object_detection(s)
                checkCollisions(s)
            i = i+1


# main function
if __name__ == "__main__":
    start_from_here()
