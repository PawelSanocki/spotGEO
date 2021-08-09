import numpy as np
import cv2
from typing import List
import myFilters.myFilters as myFilters
import matplotlib.pyplot as plt
import scipy.stats as st
import json
import os

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640

OUTER = -1.0
NEUTRAL = -1.0
INNER = - (OUTER * 24 + NEUTRAL * 16) / 9 / 1.0
CENTER = - (OUTER * 24 + NEUTRAL * 16) / 9 / 1.0
THRESH = 10
LOW_THRESH = 5
HIGH_THRESH = 20

O = OUTER
N = NEUTRAL
I = INNER
C = CENTER

def filter_image(img):
    max_blur_img = np.array(myFilters.max_blur(img,13,9))
    img = img.astype(np.int)
    max_blur_img = max_blur_img.astype(np.int)
    img = img - max_blur_img
#########################
    # kernel = get_filter(C,I,N,O)
    # kernel = gkern(7, 1) - 0.01
    # img = cv2.filter2D(img, 8, kernel, img)
#########################
    # img = template_matching_filter(img, num_matches=20, matching_method=1, conclusion_method='median')
#########################
    new_image = np.zeros_like(img)
    amount = 500
    flat = img.flatten()
    ind = np.argpartition(flat, -amount)[-amount:]

    th = flat[ind].min() - 0
    new_image[img > th] = 255
    img[img <= th] = 0

    new_image = new_image.astype(np.uint8)
    return new_image

def get_filter(C,I,N=0,O=0,K=0):
    kernel = np.array([[K,K,K,K,K,K,K,K,K],
                       [K,O,O,O,O,O,O,O,K],
                       [K,O,N,N,N,N,N,O,K],
                       [K,O,N,I,I,I,N,O,K],
                       [K,O,N,I,C,I,N,O,K],
                       [K,O,N,I,I,I,N,O,K],
                       [K,O,N,N,N,N,N,O,K],
                       [K,O,O,O,O,O,O,O,K],
                       [K,K,K,K,K,K,K,K,K]])
    return kernel


def gkern(size=5, nsig=3):
    x = np.linspace(-nsig, nsig, size+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

def template_matching_filter(img, num_matches = 10, matching_method = 0, conclusion_method = 'max'):
    methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
            cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
    results = []
    path = "model/dataset_nn/1/"
    n_files = len(next(os.walk(path))[2])
    rands = np.random.randint(n_files * 3 // 4, n_files, num_matches)
    for i in range(num_matches):
        template = cv2.imread(path + str(rands[i])+ ".png", cv2.IMREAD_GRAYSCALE)
        templated = cv2.matchTemplate(img.copy(), template, methods[matching_method])
        results.append(templated)
    results = np.array(results)
    try:
        maxi = np.max(results)
        mini = np.min(results)
    except:
        print(rands)
        print(results.shape)
        exit()

    if maxi == mini: maxi += 1
    results = 255 * (results - mini) / (maxi - mini)

    if methods[matching_method] == cv2.TM_SQDIFF or methods[matching_method] == cv2.TM_SQDIFF_NORMED:
        results = np.abs(255 - results)
    if conclusion_method == 'max':
        results = np.max(results, axis=0)
    elif conclusion_method == 'median':
        results = np.median(results, axis = 0)
    elif conclusion_method == 'percentile':
        results = np.percentile(results, 80, axis=0)
    
    results = np.pad(results, template.shape[0]//2)
    return results

def get_objects_coords(sequence, frame):
    train_anno_path = "train_anno.json"

    with open(train_anno_path, 'r') as f:
        anno = json.load(f)
    anno = anno[(sequence - 1) * 5 + frame - 1] # starts from 0
    object_coords = np.array(anno['object_coords'])
    return object_coords.astype(np.int)

def show_marked_image(img, object_coords, name):
    marked_img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    for coords in object_coords:
        # print(coords)
        marked_img = cv2.circle(marked_img, coords, 14, (0,0,255), 1)
    cv2.namedWindow(name,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name,640,480)
    cv2.imshow(name, marked_img)

if __name__ == "__main__":
    from os.path import join,realpath,abspath
    from pathlib import Path
    import tensorflow as tf
    from  model import settings
    from itertools import product
    from filter_NN import filter_NN
    WINDOW_SIZE = settings.WINDOW_SIZE
    def get_window(img, x, y, i):
        image = np.pad(img, WINDOW_SIZE//2)
        tup = (i, x, y)
        x += WINDOW_SIZE//2
        y += WINDOW_SIZE//2
        to_pred = image[x-WINDOW_SIZE//2:x+WINDOW_SIZE//2+1,y-WINDOW_SIZE//2:y+WINDOW_SIZE//2+1].reshape((WINDOW_SIZE,WINDOW_SIZE,1))
        return to_pred, tup
    model_time = "628130022_768"
    # model_time = "626144688_942"
    model = tf.keras.models.load_model('model\models\model' + str(model_time), compile=False)
    model.compile()
    for i in range(1,20): # which sequences
        for j in range(1,6): # which frames
            object_coords = get_objects_coords(i, j)
            # f, axarr = plt.subplots(1,3)
            path = Path(join(Path(__file__).parent.absolute(),"train"))
            org_img = cv2.imread(str(join(path, str(i), str(j))) + '.png', cv2.IMREAD_GRAYSCALE)
            # axarr[0].imshow(org_img)
            # axarr[0].set_title("Original")
            cv2.namedWindow('org',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('org',640,480)
            cv2.imshow('org', org_img)
            show_marked_image(org_img, object_coords, "marked_org")

            img = filter_image(org_img)

            # axarr[1].imshow(img)
            # axarr[1].set_title("Initial filter")
            cv2.namedWindow("filtered",cv2.WINDOW_NORMAL)
            cv2.resizeWindow("filtered",640,480)
            cv2.imshow("filtered", img)
            show_marked_image(img, object_coords, "marked_filtered")
            # cv2.waitKey()

            img = img / 255.0
            to_pred = []
            coords = []
            for x, y in product(range(img.shape[0]),range(img.shape[1])):
                if img[x,y] > 0:
                    window, coord = get_window(img, x, y, 0)
                    to_pred.append(window)
                    coords.append(coord)
            if len(to_pred) > 0:
                p = model.predict(np.array(to_pred))
                for it, coord in enumerate(coords):
                    img[coord[1], coord[2]] = p[it][0]
            img *= 255.
            img = img.astype(np.uint8)
            cv2.namedWindow("predicted",cv2.WINDOW_NORMAL)
            cv2.resizeWindow("predicted",640,480)
            cv2.imshow("predicted", img)
            show_marked_image(img, object_coords, "marked_predicted")
            # axarr[2].imshow(img)
            # axarr[2].set_title("Classifier")

            original_imgs = []
            for it in range(5):
                img = cv2.imread(join(path, str(i), str(it+1)) +
                                '.png', cv2.IMREAD_GRAYSCALE)
                original_imgs.append(img)
            original_imgs = np.stack(original_imgs)

            imgs, model = filter_NN(original_imgs, model)
            cv2.namedWindow("predicted2",cv2.WINDOW_NORMAL)
            cv2.resizeWindow("predicted2",640,480)
            cv2.imshow("predicted2", imgs[j-1]*255.)

            plt.show()
            cv2.waitKey(5000)
            cv2.destroyAllWindows()
            exit()


