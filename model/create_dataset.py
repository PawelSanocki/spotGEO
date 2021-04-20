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

SEED = settings.SEED
WINDOW_SIZE = settings.WINDOW_SIZE

train_anno_path = "train_anno.json"
images_path = "train"

with open(train_anno_path, 'r') as f:
    ds_frames = json.load(f)
# print(ds_frames[0])
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

# def get_false_image(anno):
#     seq = np.random.randint(1,1281)
#     frame = np.random.randint(1,6)
#     path = images_path + "\\" + str(seq) +  "\\" + str(frame) + ".png"
#     image = cv2.imread(path, 0) # 0 for greyscale
#     img = filter_image(image)
#     y, x = np.nonzero(img)
#     if len(x) > 0:
#         rand = np.random.randint(0,len(x))
#         x = x[rand]
#         y = y[rand]
#     else:
#         x = np.random.randint(0,image.shape[1])
#         y = np.random.randint(0,image.shape[0])
#     size = WINDOW_SIZE // 2
#     image = np.pad(image, size, mode = 'reflect')
#     img = image[y - size + size:y + size+1+ size, x - size+ size:x + size+1+ size] # add size everywhere due to padding
#     return img

PATH_TRUE = "model\\dataset_nn\\1\\"
for i in tqdm(range(len(ds_true))):
    img = get_image(ds_true[i], PATH_TRUE, i)
    # cv2.imwrite(PATH_TRUE + str(i) + ".png", img)

from filters import filter_image
from itertools import product

def create_false_sample(true, iter, path):
    PATH_FALSE = "model\\dataset_nn\\0\\"
    image = cv2.imread(path, 0)
    filtered_img = filter_image(image)
    size = WINDOW_SIZE // 2
    image = np.pad(image, size, mode = 'reflect')
    for y, x in product(range(filtered_img.shape[0]),range(filtered_img.shape[1])):
        if (y,x) in true:
            continue
        if filtered_img[y, x] > 0:
            iter += 1
            # if iter % 200 > 0:
            if iter % 100 > 0:
                continue
            img = image[y - size + size:y + size+1+ size, x - size+ size:x + size+1+ size] # add size everywhere due to padding
            cv2.imwrite(PATH_FALSE + str(iter) + ".png", img)
            # plt.imshow(img)
            # plt.show()
    return iter

PATH_FALSE = "model\\dataset_nn\\0\\"
# for i in tqdm(range(len(ds_true))):
#     img = get_false_image(ds_true[i])
#     cv2.imwrite(PATH_FALSE + str(i) + ".png",img)
path = "train"
iter = 0
for i in tqdm(range(len(next(os.walk(path))[1]))):
    for j in range(5):
        img_path = path + "\\" + str(i + 1) + "\\" + str(j + 1) + ".png"
        true = ((int(y), int(x)) for y, x in ds_frames[5*i + j]['object_coords'])
        iter = create_false_sample(true, iter, img_path)
print(iter)
