from itertools import product

import tensorflow as tf
from filters import filter_image
import numpy as np
from model import settings

WINDOW_SIZE = settings.WINDOW_SIZE

def get_window(img, x, y, i):
        image = np.pad(img, WINDOW_SIZE//2)
        coord = (i, x, y)
        x += WINDOW_SIZE//2
        y += WINDOW_SIZE//2
        to_pred = image[x-WINDOW_SIZE//2:x+WINDOW_SIZE//2+1,y-WINDOW_SIZE//2:y+WINDOW_SIZE//2+1].reshape((WINDOW_SIZE,WINDOW_SIZE,1))
        return to_pred, coord

def filter_NN(imgs, model = None):
    if model is None:
        model_time = 624290112
        model = tf.keras.models.load_model('model\models\model' + str(model_time), compile=False)
        model.compile()
    to_pred = []
    coords = []
    for i in range(5):
        img = imgs[i].copy()
        img = filter_image(img)
        for x, y in product(range(img.shape[0]),range(img.shape[1])):
            if img[x,y] > 0:
                window, coord = get_window(imgs[i], x, y, i)
                to_pred.append(window)
                coords.append(coord)
    filtered_images = np.zeros_like(imgs)
    if len(to_pred) > 0:
        p = model.predict(np.array(to_pred))
        for i, coord in enumerate(coords):
            filtered_images[coord] = p[i][0]
    return filtered_images
