import numpy as np
import cv2
from pathlib import Path
from os.path import join,realpath,abspath
from json import dumps
import timeit
import copy
from matplotlib import pyplot as plt

def preprocess_images(imgs: np.ndarray, satellite_th) -> np.ndarray:
    '''
    Preprocessing required before creating trajectories
    Shrinking the blobs to single pixels
    :param imgs: stacked images in the sequence, shape (5,480,640)
    :returns: stacked images in the sequence, shape (5,480,640)
    '''
    for i in range(5):
        imgs[i] = remove_blobs(imgs[i], satellite_th)
    return imgs
def remove_blobs(img, satellite_th):
    '''
    Preprocessing required before creating trajectories
    Shrinking the blobs to single pixels
    :param imgs: stacked images in the sequence, shape (5,480,640)
    :returns: stacked images in the sequence, shape (5,480,640)
    '''
    reg, mask = cv2.threshold(img, satellite_th, 1, cv2.THRESH_BINARY)
    mask = np.uint8(mask * 255)
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE) # comment img
    new_img = np.zeros_like(img)
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            cX = int(c[0,0,1])
            cY = int(c[0,0,0])
        else:
            cY = int(M["m10"] / M["m00"])
            cX = int(M["m01"] / M["m00"])
        mask = np.zeros_like(img, np.uint8)
        mask = cv2.drawContours(mask, [c], -1, 255, -1)
        _, max_val, _, _ = cv2.minMaxLoc(img, mask = mask)
        new_img[cX,cY] = max_val
    # cv2.imshow("", img)
    # cv2.waitKey()
    return new_img

def add_noise(imgs, amount=0):
    '''
    Any preprocessing required, will be adjusted to fit the results from other methods
    '''
    for i in range(5):
        for noise in range(amount):
            imgs[i, np.random.randint(imgs.shape[1]), np.random.randint(imgs.shape[2])] = 1
        imgs[i] = imgs[i]
    return imgs

def get_sequence(path: str):
    '''
    Gets sequence from given path
    '''
    imgs = []
    for i in range(5):
        imgs.append(cv2.imread(join(path, str(i+1)) + '.png', cv2.IMREAD_GRAYSCALE))
        imgs[i] = imgs[i]
    imgs = np.stack(imgs)
    return imgs

def generate_points(imgs, satellite_th):
    '''
    imgs - matrix of shape (5, 480, 640)
    '''
    
    points = []
    for t in range(5): # might be range(4) or range(5), speedup or more points to test
        points_in_frame = []
        for x in range(imgs.shape[1]):
            for y in range(imgs.shape[2]):
                # threshold, states where we are sure that model predicted satellite correctly
                if imgs[t,x,y] > satellite_th:
                    # x,y swaped due to json notation
                    points_in_frame.append(tuple((t,y,x)))
        points.append(points_in_frame)
    return points

def calculate_direction(points, imgs, score_threshold, margin):
    '''
    Finds colinear points on one image
    Returns dictionary with all directions found
    '''
    directions = dict()
    for i in range(4):
        for j in range(i+1, 5):
            for p1 in points[i]:
                for p2 in points[j]:
                    trajectory, score, direction = predict_points(p1, p2, imgs, margin)
                    if score < score_threshold: continue
                    direction = tuple(direction)
                    if direction in directions.keys():
                        directions[direction].append(copy.deepcopy(trajectory))
                    else:
                        directions[direction] = [copy.deepcopy(trajectory)]
                    break
    return directions

def predict_points(p1,p2,imgs, margin):
    p1 = np.array(p1)
    p2 = np.array(p2)
    direction = ((p1 - p2) / (p1[0] - p2[0]))
    p0 = p1 - p1[0] * direction
    trajectory = []
    for i in range(5):
        trajectory.append(tuple((p0 + direction * i).astype(int)))
    score = 0
    for p in trajectory:
        #if 0 <= p[1] < imgs.shape[1] and 0 <= p[2] < imgs.shape[2]:
        if 0 <= p[2] < imgs.shape[1] and 0 <= p[1] < imgs.shape[2]:
            #score += np.max(imgs[p[0], max(p[1]-1,0):p[1]+2, max(p[2]-1,0):p[2]+2])
            score += np.max(imgs[p[0],max(p[2]-margin,0):p[2]+1+margin , max(p[1]-margin,0):p[1]+1+margin])
        else:
            score = 0
            break
    trajectory.append(score)
    return trajectory, score, direction

def merge_images(imgs):
    return imgs[0] + imgs[1] + imgs[2] + imgs[3] + imgs[4]

def clear_trajectories(trajectories: dict, star_direction: tuple = None, keep_only_best: bool = False, 
                        directions_similarity = 10, trajectory_similarity = 10, max_trajectories = 15) -> dict:
    clean_traj = dict()
    # merge directions and directories
    for key in trajectories.keys():
        is_different = True
        for key2 in clean_traj.keys():
            if all(np.abs(np.array(key) - np.array(key2)) < directions_similarity):
                is_different = False
                for item in trajectories[key]:
                    already_included = False
                    for item2 in clean_traj[key2]:
                        if all(np.abs(np.array(item[0]) - np.array(item2[0])) < trajectory_similarity):
                            already_included = True
                            if item2[-1] < item[-1]:
                                item2 = item
                    if not already_included:
                        clean_traj[key2].append(item)
        if is_different:
            clean_traj[key] = []
            for item in trajectories[key]:
                already_included = False
                for item2 in clean_traj[key]:
                    if all(np.array(item[0]) - np.array(item2[0]) < trajectory_similarity):
                        already_included = True
                        if item2[-1] < item[-1]:
                            item2 = item
                if not already_included:
                    clean_traj[key].append(item)
    # remove direction similar to stars direction
    if star_direction != None:
        k = None
        for key in clean_traj.keys():
            if all(np.abs(np.array(star_direction) - np.array(key)) < trajectory_similarity):
                k = key
                break
        if k is not None:
            del clean_traj[k]
    # keep only one best direction
    if keep_only_best:
        biggest_number_of_trajectories = 0
        best_key = None
        for key in clean_traj.keys():
            if biggest_number_of_trajectories < len(clean_traj[key]):
                biggest_number_of_trajectories = len(clean_traj[key])
                best_key = key
        if best_key != None:
            temp_traj = dict()
            temp_traj[best_key] = clean_traj[best_key]
            clean_traj = dict(temp_traj)
    # keep up to 30 best trajectories
    while True:
        number_of_trajectories = 0
        worst_score = 5.1
        for k in clean_traj.keys():
            number_of_trajectories += len(clean_traj[k])
            for i, t in enumerate(clean_traj[k]):
                if t[-1] < worst_score:
                    worst_key = k
                    worst_score = t[-1]
                    worst_trajectory = i
        if number_of_trajectories <= max_trajectories:
            break
        clean_traj[worst_key].pop(worst_trajectory)
    return clean_traj

def find_stars_direction(imgs: np.ndarray, satellite_th = 0.5, score_threshold = 3, margin = 5) -> tuple:
    """
    Return the direction of stars shift. In case of too many detected stars, returns (100,0,0) as a direction without calculation not to increase the time.
    :param imgs: 3D, stacked original images, values 0-1 so divide by 255.0
    :return: Tuple with the direction of stars shift.
    """
    direction = (100,0,0)
    imgs = copy.deepcopy(imgs)
    # imgs = preprocess_images(imgs)
    min_max = 1
    for img in imgs:
        maximum = np.max(img)
        if maximum < min_max:
            min_max = maximum
    for img in imgs:
        _, img = cv2.threshold(img, min_max * 0.95, 1, cv2.THRESH_BINARY)
        #img = cv2.adaptiveThreshold(np.uint8(img*255), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 1,) / 255.0
    points = generate_points(imgs, satellite_th)
    for p in points:
        if len(p) > 150:
            return direction
    # for i in range(5):
    #     cv2.imshow(str(i),imgs[i])
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    trajectories = calculate_direction(points, imgs, score_threshold, margin)
    for k in trajectories.keys():
        direction = k
        break
    for k in trajectories.keys():
        if len(trajectories[direction]) < len(trajectories[k]):
            direction = k
    # print()
    # print(direction)
    return direction

def sequence_into_trajectories(imgs: np.ndarray, original_images: np.ndarray = None, keep_only_best: bool = False, 
                                satellite_th = 0.7, score_threshold = 3, margin = 5, directions_similarity = 10, 
                                trajectory_similarity = 10, max_trajectories = 15, preprocess = True) -> dict:
    '''
    Return dictionary with directions as keys and list of sequences of points
    :param imgs: 3D, stacked filteres images, values 0-1 so divide by 255.0
    :param original_images: 3D, stacked original images, values 0-1 so divide by 255.0
    :param keep_only_best: if true keeps only the direction with the biggest number of trajectories
    :return: dictionary with directions as keys and lists of trajectories as values
    '''
    if preprocess:
        imgs_clear = preprocess_images(imgs, satellite_th)
    else: imgs_clear = imgs.copy()
    # amount = 150
    # flat = imgs.flatten()
    # ind = np.argpartition(flat, -amount)[-amount:]
    # satellite_th = max(flat[ind].min(), satellite_th)
    points = generate_points(imgs_clear, satellite_th)
    for p in points:
        if len(p) > 150:
            print()
            print(len(p))
            return dict()
    trajectories = calculate_direction(points, imgs, score_threshold, margin)
    if original_images is not None:
        trajectories = clear_trajectories(trajectories, find_stars_direction(original_images), keep_only_best, directions_similarity, trajectory_similarity, max_trajectories)
    else:
        trajectories = clear_trajectories(trajectories, keep_only_best=keep_only_best, directions_similarity = directions_similarity, trajectory_similarity = trajectory_similarity, max_trajectories=max_trajectories)
    return trajectories

if __name__ == "__main__":
    task_number = 1
    path = join(Path(__file__).parent.absolute(),"train",str(task_number))
    imgs = get_sequence(path)
    # imgs = add_noise(imgs, 10)
    # img = merge_images(imgs)
    from filter_NN import filter_NN
    # cv2.imshow("original", imgs[0])
    imgs, model = filter_NN(imgs)
    # cv2.imshow("filtered", imgs[0] * 255)

    satellite_th = 0.5
    imgs_clear = preprocess_images(imgs, satellite_th)
    # cv2.imshow("blobs cleared", imgs_clear[0] * 255)
    # cv2.waitKey()

    points = generate_points(imgs_clear, satellite_th)
    trajectories = calculate_direction(points, imgs, score_threshold=3, margin=10)
    print(len(trajectories))
    trajectories = clear_trajectories(trajectories, keep_only_best=False, directions_similarity = 10, trajectory_similarity = 10, max_trajectories=15)
    print((trajectories.values()))
    from dict_to_json import label_frame
    print(label_frame(trajectories, 1, 1))
    