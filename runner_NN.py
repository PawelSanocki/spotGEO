from os.path import join,realpath,abspath
import cv2
import numpy as np
from pathlib import Path
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
from model import settings

WINDOW_SIZE = settings.WINDOW_SIZE

model_time = 7622941
model = tf.keras.models.load_model('model\models\model' + str(model_time))
model.summary()

def get_window(img, x, y, i):
    image = np.pad(img, WINDOW_SIZE//2)
    tup = (i, x, y)
    x += WINDOW_SIZE//2
    y += WINDOW_SIZE//2
    to_pred = image[x-WINDOW_SIZE//2:x+WINDOW_SIZE//2+1,y-WINDOW_SIZE//2:y+WINDOW_SIZE//2+1].reshape((WINDOW_SIZE,WINDOW_SIZE,1))
    return to_pred, tup
def print_images(original_imgs, imgs, columns = 5):
    fig=plt.figure(figsize=(14, 8))
    rows = 2
    iter = 1
    for img in original_imgs[:columns]:
        fig.add_subplot(rows, columns, iter)
        plt.imshow(img)
        iter += 1
    for img in imgs[:columns]:
        fig.add_subplot(rows, columns, iter)
        plt.imshow(img)
        iter += 1
    plt.show()

path = join(Path(__file__).parent.absolute(),"train")
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
    # print_images(original_imgs, imgs, 2)
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
    precision, recall, F1, mse = validation.compute_score('submission.json', 'train_anno.json')
    print('precision, recall, F1, mse')
    print(precision, recall, F1, mse)
