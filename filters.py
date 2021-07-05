import numpy as np
import cv2
from typing import List
#from skimage.measure import label, regionprops
import copy
# from filters import maxBlur
from scipy import stats
#import pyximport; pyximport.install()
import myFilters.myFilters as myFilters
import matplotlib.pyplot as plt
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640

OUTER = -1.0
NEUTRAL = -1.0
INNER = - (OUTER * 24 + NEUTRAL * 16) / 9 / 1.0
CENTER = - (OUTER * 24 + NEUTRAL * 16) / 9 / 1.4
THRESH = 10
LOW_THRESH = 5
HIGH_THRESH = 20

O = OUTER
N = NEUTRAL
I = INNER
C = CENTER

def filter_image(img):
    # max_blur_img = np.array(myFilters.max_blur(img,7,5))
    # max_blur_img = cv2.subtract(img,max_blur_img)
    # _, max_blur_img = cv2.threshold(max_blur_img, 1, 255, cv2.THRESH_BINARY)
    # # kernel = np.uint8(np.array([[1,1,1],[1,1,1],[1,1,1]]))
    # # max_blur_img = cv2.dilate(max_blur_img, kernel, iterations=2)

    # # img = img * max_blur_img
    # img = max_blur_img
######################################################
    # max_blur_img = np.array(myFilters.max_blur_corners(img,7))
    # max_blur_img = cv2.subtract(img,max_blur_img)
    # _, max_blur_img = cv2.threshold(max_blur_img, 10, 255, cv2.THRESH_BINARY)
    # # kernel = np.uint8(np.array([[1,1,1],[1,1,1],[1,1,1]]))
    # # max_blur_img = cv2.dilate(max_blur_img, kernel, iterations=2)

    # # img = img * max_blur_img
    # img = max_blur_img
########################################
    # max_blur_img = np.array(myFilters.max_blur(img,7,5))
    # max_blur_img = cv2.subtract(img,max_blur_img)
    # _, max_blur_img = cv2.threshold(max_blur_img, 5, 255, cv2.THRESH_BINARY)

    # img = max_blur_img
##################
    max_blur_img = np.array(myFilters.max_blur(img,13,9))
    img = img.astype(np.int)
    max_blur_img = max_blur_img.astype(np.int)
    img = img - max_blur_img
######################################## STEP by step removal
    # # find areas of 9 pixels
    # kernel = get_filter(0.3,0.3,0,-0.08,-0.03)
    # img = cv2.filter2D(img, 8, kernel, img)
    # img = np.uint8(cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)//1)
    # img = cv2.filter2D(img, 8, kernel, img)
    # img = np.uint8(cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)//1)
    # _, img = cv2.threshold(img, 1, 255, cv2.THRESH_TOZERO)
    
    # # # remove where area is too big
    # kernel = get_filter(1,0,0,-0.5,-0.5)
    # img = cv2.filter2D(img, 8, kernel, img)
    # img = np.uint8(cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)//1)
    # _, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)

    # # leave only where more than 2 pixels are next to each other
    # mask = img.copy()
    # _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    # kernel = get_filter(0.2,0.01,0,0)
    # mask = cv2.filter2D(mask, 8, kernel, mask)
    # _, mask = cv2.threshold(mask, 0.21 * 255, 255, cv2.THRESH_BINARY)
    # img = np.uint8(mask / 255.0 * img)

    # kernel = np.uint8(np.array([[0,1,0],[1,1,1],[0,1,0]]))
    # img = cv2.dilate(img, kernel, iterations=1)
############################################################################ best result so far, filter + max_blur
    # max_mask = max_blur(img,7,5)
    # img1 = cv2.subtract(img,max_mask)
    # #img1 = cv2.subtract(img1,1)
    # _, img = cv2.threshold(img1, 1, 255, cv2.THRESH_BINARY)

    # O = -1.0
    # N = -1.0
    # I = - (OUTER * 24 + NEUTRAL * 16) / 9 / 1.0
    # C = - (OUTER * 24 + NEUTRAL * 16) / 9 / 1.4
    # kernel = get_filter(C,I,N,O)
    # img = cv2.filter2D(img, 8, kernel, img)
    # _, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    
    # masked = img.copy()
    # label_img = label(masked)
    # regions = regionprops(label_img)
    # img = remove_noise(img=img,regions=regions,low_thresh=5,high_thresh=20)

    # masked = np.uint8(img.copy())
    # kernel = np.uint8(np.array([[0,1,0],[1,1,1],[0,1,0]]))
    # masked = cv2.dilate(masked, kernel, iterations=1)
    # label_img = label(masked)
    # regions = regionprops(label_img)
    # img = remove_noise(img=img,regions=regions,low_thresh=5,high_thresh=50)

    # img = cv2.bitwise_and(img,img1)
########################################################################### MAX_BLUR, filter, adaptive
    # O = -1.0
    # N = -1.0
    # I = - (OUTER * 24 + NEUTRAL * 16) / 9 / 1.0
    # C = - (OUTER * 24 + NEUTRAL * 16) / 9 / 1.37
    # LOW_THRESH = 5
    # HIGH_THRESH = 20
    # kernel = get_filter(C,I,N,O)
    # img2 = cv2.filter2D(img, 8, kernel)
    # _, img2 = cv2.threshold(img2, 10, 255, cv2.THRESH_BINARY)
    
    # masked = img2.copy()
    # label_img = label(masked)
    # regions = regionprops(label_img)
    # img2 = remove_noise(img=img2,regions=regions,low_thresh=LOW_THRESH,high_thresh=5000)

    # masked = np.uint8(img2.copy())
    # kernel = np.uint8(np.array([[0,1,0],[1,1,1],[0,1,0]]))
    # masked = cv2.dilate(masked, kernel, iterations=1)
    # label_img = label(masked)
    # regions = regionprops(label_img)
    # img2 = remove_noise(img=img2,regions=regions,low_thresh=LOW_THRESH,high_thresh=50)

    # max_mask = max_blur(img,7,5)
    # img1 = cv2.subtract(img,max_mask)
    # #img1 = cv2.subtract(img1,1)
    # _, img1 = cv2.threshold(img1, 1, 255, cv2.THRESH_BINARY)

    # img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, -5)
    # mask = img.copy()
    # _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    # kernel = get_filter(0.2,0.01,0,0)
    # mask = cv2.filter2D(mask, 8, kernel, mask)
    # _, mask = cv2.threshold(mask, 0.21 * 255, 255, cv2.THRESH_BINARY)
    # img = np.uint8(mask / 255.0 * img)
    
    # # cv2.namedWindow("img",cv2.WINDOW_NORMAL)
    # # cv2.resizeWindow("img",640,480)
    # # cv2.imshow("img", img)
    # # cv2.namedWindow("max",cv2.WINDOW_NORMAL)
    # # cv2.resizeWindow("max",640,480)
    # # cv2.imshow("max", img1)
    # # cv2.namedWindow("filter",cv2.WINDOW_NORMAL)
    # # cv2.resizeWindow("filter",640,480)
    # # cv2.imshow("filter", img2)

    # img = cv2.bitwise_and(img,img1)
    # img = cv2.bitwise_and(img,img2)

################################################################################ max_blur większy
    # max_mask = max_blur(img,7,5)
    # img1 = cv2.subtract(img,max_mask)
    # _, img = cv2.threshold(img1, 2, 255, cv2.THRESH_BINARY)
############################################################
    # max_mask = max_blur(img,7,5)
    # img1 = cv2.subtract(img,max_mask)
    # _, img1 = cv2.threshold(img1, 1, 255, cv2.THRESH_BINARY)

    # img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, -6)
    # mask = img.copy()
    # _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    # kernel = get_filter(0.2,0.01,0,0)
    # mask = cv2.filter2D(mask, 8, kernel, mask)
    # _, mask = cv2.threshold(mask, 0.21 * 255, 255, cv2.THRESH_BINARY)
    # img = np.uint8(mask / 255.0 * img)


    # img = cv2.bitwise_and(img,img1)


############################################################
    #img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, -5)
##############################################################


    amount = 1000
    flat = img.flatten()
    ind = np.argpartition(flat, -amount)[-amount:]

    th = flat[ind].min() - 0
    img[img > th] = 255
    img[img <= th] = 0

    img = img.astype(np.uint8)
    return img

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

def remove_noise(img: np.ndarray, regions: List, low_thresh: int, high_thresh: int) -> np.ndarray:
    """
    Function to remove regions that were too small, according to regionprops method
    :param img (np.ndarray): image which will be proceded later. From this picture noise is removed
    :param regions (List): List of regions found by regionprops
    :param thresh (int): Minimum size of a region. Smaller that this will be removed
    :return image with removed regions smaller than thresh area
    """
    img_zeros = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
    for props in regions:
        if high_thresh > props.area > low_thresh:
            min_row,min_col,max_row,max_col = props.bbox
            img_zeros[min_row: max_row + 1, min_col: max_col + 1] = 255
    img[img_zeros != 255] = 0
    return img

def max_blur(image, size, mask_size = 3):
    imgs = []
    for i in range(size):
        for j in range(size):
            if i < mask_size//2 or j < mask_size//2 or i > size-mask_size//2-1 or j > size-mask_size//2-1:
                kernel = np.zeros((size,size))
                kernel[i,j] = 1
                imgs.append(cv2.filter2D(image,8,kernel))
    img = np.stack(imgs,-1)
    img = np.max(img,-1)
    return img

import json
def get_objects_coords(sequence, frame):
    train_anno_path = "train_anno.json"
    images_path = "train"

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
    # model_time = "624991879_928"
    model_time = "625143927_940"
    model = tf.keras.models.load_model('model\models\model' + str(model_time), compile=False)
    model.compile()
    for i in range(1,12): # which sequences
        for j in range(1,2): # which frames
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

            to_pred = []
            coords = []
            for x, y in product(range(img.shape[0]),range(img.shape[1])):
                if img[x,y] > 0:
                    window, coord = get_window(img, x, y, 0)
                    to_pred.append(window)
                    coords.append(coord)
            img = img / 255.0
            if len(to_pred) > 0:
                p = model.predict(np.array(to_pred))
                for i, coord in enumerate(coords):
                    img[coord[1], coord[2]] = p[i][0]
            img *= 255.
            img = img.astype(np.uint8)
            cv2.namedWindow("predicted",cv2.WINDOW_NORMAL)
            cv2.resizeWindow("predicted",640,480)
            cv2.imshow("predicted", img)
            show_marked_image(img, object_coords, "marked_predicted")
            # axarr[2].imshow(img)
            # axarr[2].set_title("Classifier")
            plt.show()
            cv2.waitKey()
            cv2.destroyAllWindows()


