from itertools import product

import tensorflow as tf
from filters import filter_image
import numpy as np
from model import settings

WINDOW_SIZE = settings.WINDOW_SIZE

def get_window(img, x, y, i):
    '''
    Extracting window from the image to be fed to the classifier
    x, y, i - coordinates of the window
    img - the image to be filtered
    '''
    image = np.pad(img, WINDOW_SIZE//2)
    coord = (i, x, y)
    x += WINDOW_SIZE//2
    y += WINDOW_SIZE//2
    to_pred = image[x-WINDOW_SIZE//2:x+WINDOW_SIZE//2+1,y-WINDOW_SIZE//2:y+WINDOW_SIZE//2+1].reshape((WINDOW_SIZE,WINDOW_SIZE,1))
    return to_pred, coord

def filter_NN(imgs, model = None):
    '''
    Method for image filtering using R_CNN based approach
    '''
    if model is None:
        model_time = "628130022_768"
        model = tf.keras.models.load_model('model\models\model' + str(model_time), compile=False)
        model.compile()
    to_pred = []
    coords = []
    filtered_images = np.zeros_like(imgs)
    for i in range(5):
        f_img = filter_image(imgs[i].copy()) / 255.
        for x, y in product(range(f_img.shape[0]),range(f_img.shape[1])):
            if f_img[x,y] > 0:
                window, coord = get_window(f_img, x, y, i)
                to_pred.append(window)
                coords.append(coord)
    if len(to_pred) > 0:
        p = model.predict(np.array(to_pred))
        for it, coord in enumerate(coords):
            filtered_images[coord] = p[it][0]
    return filtered_images, model
