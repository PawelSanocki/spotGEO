from os import join,realpath,abspath
import numpy as np
from pathlib import Path
import cv2
from filters import filter_image
from trajectories2 import sequence_into_trajectories
from dict_to_json import label_frame
import json
import os
from tqdm import tqdm
import validation
import time
import tensorflow as tf
from itertools import product
import matplotlib.pyplot as plt

SIZE = 11

model_time = 7114191
model = tf.keras.models.load_model('model\models\model' + str(model_time))
model.summary()

def get_window(img, x, y, i):
    image = np.pad(img, SIZE//2)
    tup = (i, x, y)
    x += SIZE//2
    y += SIZE//2
    to_pred = image[x-SIZE//2:x+SIZE//2+1,y-SIZE//2:y+SIZE//2+1].reshape((SIZE,SIZE,1))
    return to_pred, tup

path = join(Path(__file__).parent.absolute(),"test")
# path = join(Path(__file__).parent.absolute(),"train")
results = []

# for seq in tqdm(range(1)):
for seq in tqdm(range(len(next(os.walk(path))[1]))):
    imgs = []
    original_imgs = []
    for i in range(5):
        img = cv2.imread(join(path, str(seq+1), str(i+1)) + '.png', cv2.IMREAD_GRAYSCALE)
        img = img / 255.0
        original_imgs.append(img)
    original_imgs = np.stack(original_imgs)
    to_pred = []
    coords = []
    for i in range(5):
        img = cv2.imread(join(path, str(seq+1), str(i+1)) + '.png', cv2.IMREAD_GRAYSCALE)
        img = filter_image(img)
        new_img = np.zeros_like(img)
        for x, y in product(range(img.shape[0]),range(img.shape[1])):
            if img[x,y] > 0:
                window, coord = get_window(img, x, y, i)
                to_pred.append(window)
                coords.append(coord)
        img = img / 255.0
        imgs.append(img)
    imgs = np.stack(imgs)
    if len(to_pred) > 0:
        p = model.predict(np.array(to_pred))
        for i, coord in enumerate(coords):
            imgs[coord] = p[i][0]
    d = sequence_into_trajectories(imgs, original_imgs, True)
    for i in range(5):
        results.append(label_frame(d, seq+1, i+1))

with open('submission.json', 'w') as outfile:
    outfile.write(str(results).replace("'", ""))

if path == join(Path(__file__).parent.absolute(),"test"):
    with open('submission.json', 'r') as f:
        results = json.load(f)
        print(validation.validate_json(results,False))
    precision, recall, F1, mse = validation.compute_score('submission.json', 'test_anno.json')
    print('precision, recall, F1, mse')
    print(precision, recall, F1, mse)
else:
    precision, recall, F1, mse = validation.compute_score('submission.json', 'annotation.json')
    print('precision, recall, F1, mse')
    print(precision, recall, F1, mse)
