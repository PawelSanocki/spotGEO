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
from grid_division_filter import filter as gdf

WINDOW_SIZE = settings.WINDOW_SIZE
NUM_SEQUENCES = 100


def print_images(original_imgs, imgs, columns=5):
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


def run(model=None):
    path = join(Path(__file__).parent.absolute(), "train")
    # path = join(Path(__file__).parent.absolute(),"train")
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
        

        # imgs, model = filter_NN(original_imgs, model)
        # original_imgs = original_imgs.astype(np.float) / 255.
        # d = sequence_into_trajectories(imgs, original_imgs, preprocess=True, satellite_th=0.4, score_threshold=2.5, keep_only_best=True, trajectory_similarity=5, )

        imgs, model = gdf(original_imgs, model)
        original_imgs = original_imgs.astype(np.float) / 255.
        d = sequence_into_trajectories(imgs, satellite_th=0.3, score_threshold=1.,
                                       keep_only_best=True, trajectory_similarity=5)

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
    else:
        precision, recall, F1, mse = validation.compute_score(
            'submission.json', 'test_anno.json')
        print('precision, recall, F1, mse')
        print(precision, recall, F1, mse)


if __name__ == "__main__":
    run()
