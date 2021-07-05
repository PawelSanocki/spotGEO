import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from filters import filter_image
from model import settings
import os, shutil

SEED = settings.SEED
WINDOW_SIZE = settings.WINDOW_SIZE

def create_dataset():

    train_anno_path = "train_anno.json"
    images_path = "train"

    with open(train_anno_path, 'r') as f:
        ds_frames = json.load(f)
    ds_true = []
    for frame in ds_frames:
        for sat in range(frame['num_objects']):
            ds_true.append((frame['sequence_id'], frame['frame'], int(frame['object_coords'][sat][0]), int(frame['object_coords'][sat][1])))

    from itertools import product  
    def get_image(anno, PATH_TRUE, i):
        seq = anno[0]
        frame = anno[1]
        x0 = anno[2]
        y0 = anno[3]
        it = 0
        for dx, dy in product([-1,0,1], [-1,0,1]):
            x = x0 + dx
            if x < 0 or x > 640: continue
            y = y0 + dy
            if y < 0 or y > 480: continue
            path = images_path + "\\" + str(seq) +  "\\" + str(frame) + ".png"
            image = cv2.imread(path, 0) # 0 for greyscale
            size = WINDOW_SIZE // 2
            image = np.pad(image, size, mode = 'reflect')
            img = image[y - size + size:y + size+1+ size, x - size+ size:x + size+1+ size] # add size everywhere due to padding
            cv2.imwrite(PATH_TRUE + str(i * 5 + it) + ".png", img)
            it += 1
        return img

    # PATH_TRUE = "model\\dataset_nn\\1\\"
    # for i in tqdm(range(500, len(ds_true))):
    #     img = get_image(ds_true[i], PATH_TRUE, i)
    #     cv2.imwrite(PATH_TRUE + str(i) + ".png", img)

    from itertools import product

    def create_false_sample(true, iter, path):
        PATH_FALSE = "model\\dataset_nn\\0\\"
        image = cv2.imread(path, 0)
        filtered_img = filter_image(image)
        size = WINDOW_SIZE // 2
        image = np.pad(image, size, mode = 'reflect')
        for y, x in product(range(filtered_img.shape[0]),range(filtered_img.shape[1])):
            if filtered_img[y, x] == 0: continue
            flag = False
            for dx, dy in product([-1,0,1], [-1,0,1]):
                if (y+dy,x+dx) in true:
                    flag = True
            if flag: continue
            if filtered_img[y, x] > 0:
                iter += 1
                if iter % 30 > 0:
                    continue
                img = image[y - size + size:y + size+1+ size, x - size+ size:x + size+1+ size] # add size everywhere due to padding
                cv2.imwrite(PATH_FALSE + str(iter) + ".png", img)
                # plt.imshow(img)
                # plt.show()
        return iter
    PATH_FALSE = "model\\dataset_nn\\0\\"
    for filename in os.listdir(PATH_FALSE):
        file_path = os.path.join(PATH_FALSE, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    path = "train"
    iter = 0
    for i in tqdm(range(100, len(next(os.walk(path))[1]))):
        for j in range(5):
            img_path = path + "\\" + str(i + 1) + "\\" + str(j + 1) + ".png"
            true = ((int(y), int(x)) for y, x in ds_frames[5*i + j]['object_coords'])
            iter = create_false_sample(true, iter, img_path)

if __name__ == "__main__":
    create_dataset()