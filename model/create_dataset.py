import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from filters import filter_image

SEED = 24
KERNEL_SIZE = 11
N_CHANNELS = 1
N_CLASSES = 2

train_anno_path = "train_anno.json"
images_path = "train"

with open(train_anno_path, 'r') as f:
    ds_frames = json.load(f)
# print(ds_frames[0])
ds_true = []
for frame in ds_frames:
    for sat in range(frame['num_objects']):
        ds_true.append((frame['sequence_id'], frame['frame'], int(frame['object_coords'][sat][0]), int(frame['object_coords'][sat][1])))
        
def get_image(anno):
    seq = anno[0]
    frame = anno[1]
    x = anno[2]
    y = anno[3]
    path = images_path + "\\" + str(seq) +  "\\" + str(frame) + ".png"
    image = cv2.imread(path, 0) # 0 for greyscale
    size = KERNEL_SIZE // 2
    image = np.pad(image, size)
    img = image[y - size + size:y + size+1+ size, x - size+ size:x + size+1+ size] # add size everywhere due to padding
    return img

def get_false_image(anno):
    seq = np.random.randint(1,1281)
    frame = np.random.randint(1,6)
    path = images_path + "\\" + str(seq) +  "\\" + str(frame) + ".png"
    image = cv2.imread(path, 0) # 0 for greyscale
    img = filter_image(image)
    y, x = np.nonzero(img)
    if len(x) > 0:
        rand = np.random.randint(0,len(x))
        x = x[rand]
        y = y[rand]
    else:
        x = np.random.randint(0,image.shape[1])
        y = np.random.randint(0,image.shape[0])
    size = KERNEL_SIZE // 2
    image = np.pad(image, size)
    img = image[y - size + size:y + size+1+ size, x - size+ size:x + size+1+ size] # add size everywhere due to padding
    return img

PATH_TRUE = "model\\dataset_nn\\1\\"
for i in tqdm(range(len(ds_true))):
    img = get_image(ds_true[i])
    cv2.imwrite(PATH_TRUE + str(i) + ".png",img)

PATH_FALSE = "model\\dataset_nn\\0\\"
for i in tqdm(range(len(ds_true))):
    img = get_false_image(ds_true[i])
    cv2.imwrite(PATH_FALSE + str(i) + ".png",img)