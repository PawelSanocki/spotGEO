from matplotlib.pyplot import axis
from grid_division_filter import filter as gdf
from filters import filter_image
from filter_NN import filter_NN
import json
from os.path import join
from pathlib import Path
import cv2
import numpy as np

def annotate_img(object_coords):
    '''
    object_coords - list of object coords from annotations
    '''
    marked = np.zeros((480,640, 3))
    for oc in object_coords:
        for coords in oc:
            marked = cv2.circle(marked, coords, 8, (0,0,255), 1)
    return marked

def example():
    with open("test_anno.json", 'r') as f:
        annotations = json.load(f)
        
    path = join(Path(__file__).parent.absolute(),"test")
    gdf_model = None
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

        gdf_filtered, gdf_model = gdf(original_imgs.copy(), gdf_model)
        gdf_filtered_sum = np.sum(gdf_filtered, axis = 0)
        gdf_filtered_sum[gdf_filtered_sum > 1] = 1
        gdf_filtered_sum = cv2.cvtColor(gdf_filtered_sum, cv2.COLOR_GRAY2BGR)
        gdf_filtered_sum = gdf_filtered_sum + satellite_marked
        cv2.imshow("gdf", gdf_filtered_sum)

        pipeline_filtered, model_NN = filter_NN(original_imgs, model_NN)
        pipeline_filtered_sum = np.sum(pipeline_filtered, axis = 0)
        pipeline_filtered_sum[pipeline_filtered_sum > 1] = 1
        pipeline_filtered_sum = cv2.cvtColor(np.float32(pipeline_filtered_sum), cv2.COLOR_GRAY2BGR)
        pipeline_filtered_sum = pipeline_filtered_sum + satellite_marked
        cv2.imshow("pipeline", pipeline_filtered_sum)

        cv2.waitKey()

if __name__ == "__main__":
    example()

