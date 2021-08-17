from os.path import join, realpath, abspath
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
from segmentation_deep_learning import filter as gdf

segmentation = True
NUM_SEQUENCES = 100
folder = "train" # "test"
WINDOW_SIZE = settings.WINDOW_SIZE

def print_images(original_imgs, imgs, columns=5):
    '''
    Printing of images for debugging purposes
    '''
    fig = plt.figure(figsize=(14, 8))
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


def run(model=None, satellite_th=0.1, score_threshold=1.1,
                                       keep_only_best=True, trajectory_similarity=15, 
                                       margin = 5, directions_similarity=5, it = 0, GDF = True):
    '''
    Method running the whole pipeline and evaluating the solution.
    satellite_th - satellite threshold for trajectories creation step
    keep_only_best - boolean, determines if only bes direction of satellite movement should be kept
    trajectory_similarity - TS used during trajectories erging step
    directions_similarity - DS used during directions merging step
    margin - margin used during trajectory score calculation
    it - current iteration of experiments
    GDF - boolen, if True SDL method will be used
    '''
    path = join(Path(__file__).parent.absolute(), folder)
    results = []
    for seq in tqdm(range(NUM_SEQUENCES)):
    # for seq in tqdm(range(len(next(os.walk(path))[1]))):
        # imgs = []
        original_imgs = []
        for i in range(5):
            img = cv2.imread(join(path, str(seq+1), str(i+1)) +
                             '.png', cv2.IMREAD_GRAYSCALE)
            original_imgs.append(img)
        original_imgs = np.stack(original_imgs)
        
        if GDF == False:
            imgs, model = filter_NN(original_imgs, model)
            original_imgs = original_imgs.astype(np.float) / 255.
            d = sequence_into_trajectories(imgs, satellite_th=satellite_th, score_threshold=score_threshold,
                                        keep_only_best=keep_only_best, trajectory_similarity=trajectory_similarity, 
                                        margin = margin, directions_similarity=directions_similarity)
        else:
            imgs, model = gdf(original_imgs, model)
            original_imgs = original_imgs.astype(np.float) / 255.
            d = sequence_into_trajectories(imgs, satellite_th=satellite_th, score_threshold=score_threshold,
                                        keep_only_best=keep_only_best, trajectory_similarity=trajectory_similarity, 
                                        margin = margin, directions_similarity=directions_similarity)

        for i in range(5):
            results.append(label_frame(d, seq+1, i+1))

    with open('submission.json', 'w') as outfile:
        outfile.write(str(results).replace("'", ""))

    if path == join(Path(__file__).parent.absolute(), "train"):
        # with open('submission.json', 'r') as f:
        #     results = json.load(f)
        #     print(validation.validate_json(results,False))
        precision, recall, F1, mse = validation.compute_score(
            'submission.json', 'train_anno.json')
        print('precision, recall, F1, mse')
        print(precision, recall, F1, mse)
        print(it, satellite_th, score_threshold, keep_only_best, trajectory_similarity, directions_similarity, margin, round(F1, 3), round(recall,3), round(precision, 3), round(mse, 0), sep=' & ')
    else:
        precision, recall, F1, mse = validation.compute_score(
            'submission.json', 'test_anno.json')
        print('precision, recall, F1, mse')
        print(precision, recall, F1, mse)
        print(round(F1, 3), round(recall,3), round(precision, 3), round(mse, 0), sep=' & ')


if __name__ == "__main__":
    STs = [0.9]
    TTs = [4.1]
    keep_bests = [1]
    TSs = [15]
    Ms = [5]
    DSs = [5]
    i = 1
    GDF = segmentation
    for ST in STs:
        for TT in TTs:
            for keep_best in keep_bests:
                for TS in TSs:
                    for M in Ms:
                        for DS in DSs:
                            run(None, ST,TT,keep_best, TS,M, DS, i, GDF)
                            i+=1



    
