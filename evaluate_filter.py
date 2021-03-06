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
from filter_NN import filter_NN
from segmentation_deep_learning import filter as gdf_filter

ST = 0.5 # satellite threshold
NEIGHBOUR_SIZE = 10

def remove_neighbourhood(mask : np.ndarray, y : int, x : int) -> np.ndarray:
    '''
    Method introduced not to allow marking single true satellite, by several predictions
    x, y - the coordinates of prediction
    mask - matrix of predictions
    '''
    flag = False
    for i in range(-NEIGHBOUR_SIZE, NEIGHBOUR_SIZE+1):
        for j in range(-NEIGHBOUR_SIZE, NEIGHBOUR_SIZE+1):
            if np.linalg.norm(np.array((y-i,x-j))) > NEIGHBOUR_SIZE: continue
            if not (i == j == 0 or 0 > y + i or y + i >= 480 or 0 > x + j or x + j >= 640):
                if mask[y + i, x + j] >= ST: flag = True
    if flag:
        for i in range(-NEIGHBOUR_SIZE, NEIGHBOUR_SIZE+1):
            for j in range(-NEIGHBOUR_SIZE, NEIGHBOUR_SIZE+1):
                if np.linalg.norm(np.array((y-i,x-j))) > NEIGHBOUR_SIZE: continue
                if not (i == j == 0 or 0 > y + i or y + i >= 480 or 0 > x + j or x + j >= 640):
                    mask[y + i, x + j] = 0
    return mask

def evaluate_image(mask : np.ndarray, anno : dict) -> tuple:
    '''
    Method for calculating number of TP, TN, FP, FN for predition matrix
    '''
    TP, TN, FP, FN = 0,0,0,0
    coords = anno["object_coords"]
    for coord in coords:
        if mask[int(coord[1])][int(coord[0])] >= ST:
            mask = remove_neighbourhood(mask, int(coord[1]), int(coord[0]))
            TP += 1
        else:
            FN += 1
    FP = (mask>=ST).sum() - TP
    TN = mask.size - TP - FN - FP
    return TP, TN, FP, FN
    
# def print_images(original_imgs, imgs, columns = 5):
#         fig=plt.figure(figsize=(14, 8))
#         rows = 2
#         iter = 1
#         for img in original_imgs[:columns]:
#             fig.add_subplot(rows, columns, iter)
#             plt.imshow(img)
#             iter += 1
#         for img in imgs[:columns]:
#             fig.add_subplot(rows, columns, iter)
#             plt.imshow(img)
#             iter += 1
#         plt.show()

def evaluate(model = None):
    '''
    Method for fileter evaluation before trajevtories creation step.
    If no model is specified it will use the default one with time of creation: 625143927_940
    '''
    if model is None:
        model_time = "625143927_940"
        model = tf.keras.models.load_model('model\models\model' + str(model_time), compile=False)
        model.compile()

    with open("train_anno.json", 'r') as f:
        annotations = json.load(f)
    
    path = join(Path(__file__).parent.absolute(),"train")

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    TP_NN = 0
    TN_NN = 0
    FP_NN = 0
    FN_NN = 0
    model = None
    for seq in tqdm(range(100)):
    # for seq in tqdm(range(len(next(os.walk(path))[1]))):
        original_imgs = []
        for i in range(5):
            img = cv2.imread(join(path, str(seq+1), str(i+1)) + '.png', cv2.IMREAD_GRAYSCALE)
            original_imgs.append(img)
        original_imgs = np.stack(original_imgs)
        imgs = []
        for img in original_imgs:
            image = filter_image(img) / 255.
            imgs.append(image)
        # imgs, model = gdf_filter(original_imgs, model)
        for i, img in enumerate(imgs):
            tp, tn, fp, fn = evaluate_image(img, annotations[seq*5+i])
            TP += tp
            TN += tn
            FP += fp
            FN += fn
        imgs_NN, model = filter_NN(original_imgs, model)
        for i, img in enumerate(imgs_NN):
            tp, tn, fp, fn = evaluate_image(img, annotations[seq*5+i])
            TP_NN += tp
            TN_NN += tn
            FP_NN += fp
            FN_NN += fn

    print("Results:")
    print()

    print("Filter")
    print("TP, TN, FP, FN")
    print(TP, TN, FP, FN)
    print("precision, recall")
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    print(precision, recall)
    print("F1")
    print(2*precision*recall / (precision+recall))
    print(round(2*precision*recall / (precision+recall), 3), round(precision,3), round(recall, 3), sep=' & ')
    print()

    print("NN")
    print("TP, TN, FP, FN")
    print(TP_NN, TN_NN, FP_NN, FN_NN)
    print("precision, recall")
    precision = TP_NN / (TP_NN + FP_NN)
    recall = TP_NN / (TP_NN + FN_NN)
    print(precision, recall)
    print("F1")
    print(2*precision*recall / (precision+recall))
    print(round(2*precision*recall / (precision+recall), 3), round(precision,3), round(recall, 3), sep=' & ')

if __name__ == "__main__":
    evaluate()