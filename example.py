from matplotlib.pyplot import axis
from segmentation_deep_learning import filter as sdl
from filters import filter_image
from filter_NN import filter_NN
import json
from os.path import join
from pathlib import Path
import cv2
import numpy as np
from trajectories2 import sequence_into_trajectories

def annotate_img(object_coords):
    '''
    object_coords - list of object coords from annotations
    '''
    marked = np.zeros((480,640, 3))
    for oc in object_coords:
        for coords in oc:
            marked = cv2.circle(marked, coords, 8, (0,0,255), 1)
    return marked

def get_final_imgs(d):
    imgs = np.zeros((480,640,3))
    for value in d.values():
        for traj in range(len(value)):
            for i in range(5):
                imgs = cv2.circle(imgs, (value[traj][i][1],value[traj][i][2]), 10, (0,255,0), 1)
    return imgs

def example():
    with open("test_anno.json", 'r') as f:
        annotations = json.load(f)
        
    path = join(Path(__file__).parent.absolute(),"test")
    sdl_model = None
    model_NN = None
    for seq in range(0, 5):
        original_imgs = []
        object_coords = []
        for frame in range(5):
            img = cv2.imread(join(path, str(seq+1), str(frame+1)) + '.png', cv2.IMREAD_GRAYSCALE)
            original_imgs.append(img)
            oc = np.array(annotations[(seq) * 5 + frame]['object_coords']).astype(int)
            object_coords.append(oc)
        original_imgs = np.stack(original_imgs)
        satellite_marked = annotate_img(object_coords)

        org_img = cv2.cvtColor(original_imgs[0].copy(), cv2.COLOR_GRAY2BGR)
        org_marked = annotate_img(object_coords[:len(object_coords) // 5])
        marked_img_org = org_img + org_marked
        marked_img_org[marked_img_org > 255] = 255
        cv2.imshow("Original image", marked_img_org.astype(np.uint8))

        sdl_filtered, sdl_model = sdl(original_imgs.copy(), sdl_model)
        sdl_filtered_sum = np.sum(sdl_filtered, axis = 0)
        sdl_filtered_sum[sdl_filtered_sum > 1] = 1
        sdl_filtered_sum = cv2.cvtColor(sdl_filtered_sum, cv2.COLOR_GRAY2BGR)
        sdl_filtered_sum = sdl_filtered_sum + satellite_marked
        cv2.imshow("Filtered - SDL", sdl_filtered_sum)

        d = sequence_into_trajectories(sdl_filtered, satellite_th=0.05, score_threshold=1.4,
                                        keep_only_best=1, trajectory_similarity=15, 
                                        margin = 5, directions_similarity=5)
        final_imgs  = get_final_imgs(d)
        marked_final_imgs = final_imgs + satellite_marked
        cv2.imshow("Final result - SDL", marked_final_imgs)

        pipeline_filtered, model_NN = filter_NN(original_imgs, model_NN)
        pipeline_filtered_sum = np.sum(pipeline_filtered, axis = 0)
        pipeline_filtered_sum[pipeline_filtered_sum > 1] = 1
        pipeline_filtered_sum = cv2.cvtColor(np.float32(pipeline_filtered_sum), cv2.COLOR_GRAY2BGR)
        pipeline_filtered_sum = pipeline_filtered_sum + satellite_marked
        cv2.imshow("Filtered - R-CNN based", pipeline_filtered_sum)

        d = sequence_into_trajectories(sdl_filtered, satellite_th=0.9, score_threshold=3.1,
                                        keep_only_best=1, trajectory_similarity=15, 
                                        margin = 5, directions_similarity=5)
        final_imgs  = get_final_imgs(d)
        marked_final_imgs = final_imgs + satellite_marked
        cv2.imshow("Final result - R-CNN based", marked_final_imgs)

        cv2.waitKey()

if __name__ == "__main__":
    example()

