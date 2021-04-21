import numpy as np
import cv2
from pathlib import Path
from os.path import join,realpath,abspath
from json import dumps
import timeit
import copy
from matplotlib import pyplot as plt

SATELLITE = 0.4
SCORE_THRESHOLD = 2.2
MARGIN = 3
DIRECTIONS_SIMILARITY = 10
TRAJECTORY_SIMILARITY = 10

def preprocess_images(imgs: np.ndarray) -> np.ndarray:
    '''
    Preprocessing required before creating trajectories
    Shrinking the blobs to single pixels
    :param imgs: stacked images in the sequence, shape (5,480,640)
    :returns: stacked images in the sequence, shape (5,480,640)
    '''
    for i in range(5):
        imgs[i] = remove_blobs(imgs[i])
    return imgs
def remove_blobs(img):
    '''
    Preprocessing required before creating trajectories
    Shrinking the blobs to single pixels
    :param imgs: stacked images in the sequence, shape (5,480,640)
    :returns: stacked images in the sequence, shape (5,480,640)
    '''
    reg, img = cv2.threshold(img, SATELLITE, 1, cv2.THRESH_BINARY)
    # cv2.imshow("1", img)
    # cv2.waitKey()
    img = np.uint8(img * 255)
    _, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # comment img
    img = np.zeros_like(img)
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            cX = int(c[0,0,1])
            cY = int(c[0,0,0])
        else:
            cY = int(M["m10"] / M["m00"])
            cX = int(M["m01"] / M["m00"])
        img[cX,cY] = 255
    # cv2.imshow("", img)
    # cv2.waitKey()
    return np.int32(img) / 255
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
        imgs[i] = imgs[i] / 255
    imgs = np.stack(imgs)
    return imgs

def generate_points(imgs):
    '''
    imgs - matrix of shape (5, 480, 640)
    '''
    points = []
    for t in range(5): # might be range(4) or range(5), speedup or more points to test
        points_in_frame = []
        for x in range(imgs.shape[1]):
            for y in range(imgs.shape[2]):
                # threshold, states where we are sure that model predicted satellite correctly
                if imgs[t,x,y] > SATELLITE:
                    # x,y swaped due to json notation
                    points_in_frame.append(tuple((t,y,x)))
        points.append(points_in_frame)
    return points

def calculate_direction(points, imgs):
    '''
    Finds colinear points on one image
    Returns dictionary with all directions found
    '''
    # directions = dict()
    # for i in range(4):
    #     for j in range(i+1, 5): # might be in range(i+1, 4) for speedup as we do not need to find just two points, range(i+1, 5) if we need better results
    #         points_copy_1 = copy.deepcopy(points)
    #         for p_1 in range(len(points_copy_1[i])):
    #             p1 = points_copy_1[i][p_1]
    #             points_copy_2 = copy.deepcopy(points)
    #             for p_2 in range(len(points_copy_2[j])):
    #                 p2 = points_copy_2[j][p_2]
    #                 trajectory, score, direction = predict_points(p1, p2, imgs)
    #                 if score < SCORE_THRESHOLD: continue
    #                 direction = tuple(direction)
    #                 if direction in directions.keys():
    #                     directions[direction].append(copy.deepcopy(trajectory))
    #                 else:
    #                     directions[direction] = [copy.deepcopy(trajectory)]
    #                 break
    # return directions
    directions = dict()
    for i in range(4):
        for j in range(i+1, 5): # might be in range(i+1, 4) for speedup as we do not need to find just two points, range(i+1, 5) if we need better results
            for p1 in points[i]:
                for p2 in points[j]:
                    trajectory, score, direction = predict_points(p1, p2, imgs)
                    if score < SCORE_THRESHOLD: continue
                    direction = tuple(direction)
                    if direction in directions.keys():
                        directions[direction].append(copy.deepcopy(trajectory))
                    else:
                        directions[direction] = [copy.deepcopy(trajectory)]
                    break
    return directions

def predict_points(p1,p2,imgs):
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
            score += np.max(imgs[p[0],max(p[2]-MARGIN,0):p[2]+1+MARGIN , max(p[1]-MARGIN,0):p[1]+1+MARGIN])
        else:
            score = 0
            break
    trajectory.append(score)
    return trajectory, score, direction

def merge_images(imgs):
    return imgs[0] + imgs[1] + imgs[2] + imgs[3] + imgs[4]

def clear_trajectories(trajectories: dict, star_direction: tuple = None, keep_only_best: bool = False) -> dict:
    clean_traj = dict()
    # merge directions and directories
    for key in trajectories.keys():
        is_different = True
        for key2 in clean_traj.keys():
            if all(np.abs(np.array(key) - np.array(key2)) < DIRECTIONS_SIMILARITY):
                is_different = False
                for item in trajectories[key]:
                    already_included = False
                    for item2 in clean_traj[key2]:
                        if all(np.abs(np.array(item[0]) - np.array(item2[0])) < TRAJECTORY_SIMILARITY):
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
                    if all(np.array(item[0]) - np.array(item2[0]) < TRAJECTORY_SIMILARITY):
                        already_included = True
                        if item2[-1] < item[-1]:
                            item2 = item
                if not already_included:
                    clean_traj[key].append(item)
    # remove direction similar to stars direction
    if star_direction != None:
        k = None
        for key in clean_traj.keys():
            if all(np.abs(np.array(star_direction) - np.array(key)) < TRAJECTORY_SIMILARITY):
                k = key
                break
        if k is not None:
            del clean_traj[k]
    # keep only one best direction
    if keep_only_best:
        biggest_number_of_trajectories = 0
        best_score = 0
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
            for t in clean_traj[k]:
                if t[-1] < worst_score:
                    worst_key = k
                    worst_score = t[-1]
                    worst_trajectory = t
        if number_of_trajectories <= 30:
            break
        clean_traj[worst_key].remove(worst_trajectory)
    return clean_traj

def find_stars_direction(imgs: np.ndarray) -> tuple:
    """
    Return the direction of stars shift. In case of too many detected stars, returns (100,0,0) as a direction without calculation not to increase the time.
    :param imgs: 3D, stacked original images, values 0-1 so divide by 255.0
    :return: Tuple with the direction of stars shift.
    """
    direction = (100,0,0)
    imgs = copy.deepcopy(imgs)
    imgs = preprocess_images(imgs)
    min_max = 1
    for img in imgs:
        maximum = np.max(img)
        if maximum < min_max:
            min_max = maximum
    for img in imgs:
        _, img = cv2.threshold(img, min_max * 0.95, 1, cv2.THRESH_BINARY)
        #img = cv2.adaptiveThreshold(np.uint8(img*255), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 1,) / 255.0
    points = generate_points(imgs)
    for p in points:
        if len(p) > 50:
            return direction
    # for i in range(5):
    #     cv2.imshow(str(i),imgs[i])
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    trajectories = calculate_direction(points, imgs)
    for k in trajectories.keys():
        direction = k
        break
    for k in trajectories.keys():
        if len(trajectories[direction]) < len(trajectories[k]):
            direction = k
    # print()
    # print(direction)
    return direction

def sequence_into_trajectories(imgs: np.ndarray, original_images: np.ndarray = None, keep_only_best: bool = False) -> dict:
    '''
    Return dictionary with directions as keys and list of sequences of points
    :param imgs: 3D, stacked filteres images, values 0-1 so divide by 255.0
    :param original_images: 3D, stacked original images, values 0-1 so divide by 255.0
    :param keep_only_best: if true keeps only the direction with the biggest number of trajectories
    :return: dictionary with directions as keys and lists of trajectories as values
    '''
    imgs_clear = preprocess_images(imgs)
    points = generate_points(imgs_clear)
    for p in points:
        if len(p) > 150:
            print()
            print(len(p))
            return dict()
    trajectories = calculate_direction(points, imgs)
    if original_images is not None:
        trajectories = clear_trajectories(trajectories, find_stars_direction(original_images), keep_only_best)
    else:
        trajectories = clear_trajectories(trajectories, keep_only_best=keep_only_best)
    return trajectories

if __name__ == "__main__":
    task_number = 1
    path = join(Path(__file__).parent.absolute(),"groundtruth",str(task_number))
    imgs = get_sequence(path)
    imgs = add_noise(imgs, 10)
    img = merge_images(imgs)
    start = timeit.default_timer()

    directions = sequence_into_trajectories(imgs,keep_only_best=True)

    stop = timeit.default_timer()
    print('Time: ', stop - start) 
    print("MARGIN: ", MARGIN)
    print("SCORE_THRESHOLD: ", SCORE_THRESHOLD)
    #print(directions)
    for i in directions.keys():
        print(i)
        #print(len(directions[i]))
        for satellite_trajectory in directions[i]:
            print(*satellite_trajectory)
        print()

    gts = get_sequence(path)
    gt = merge_images(gts)

    rgb_img = cv2.merge([img,img,img])
    def in_directory(d:dict, y, x):
        for k in d.keys():
            l = d[k]
            for traj in l:
                for p in range(len(traj)-1):
                    if traj[p][1] == x and traj[p][2] == y:
                        return True
        return False
    size = 6
    thickness = 1
    for x in range(gt.shape[0]):
        for y in range(gt.shape[1]):
            if gt[x,y] > 0 and in_directory(directions, x, y):
                cv2.circle(rgb_img,(y,x), size, (0, 255, 0), thickness)
            elif gt[x,y] > 0:
                cv2.circle(rgb_img,(y,x), size, (0, 0, 255),thickness)
            elif gt[x,y] == 0 and in_directory(directions, x, y):
                cv2.circle(rgb_img,(y,x), size, (255, 50, 0),thickness)
    cv2.namedWindow('',cv2.WINDOW_NORMAL)
    cv2.imshow('', rgb_img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    # for i in range(3):
    #     cv2.namedWindow(str(i),cv2.WINDOW_NORMAL)
    #     cv2.imshow(str(i), imgs[i*2])
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # plt.figure(num=None, figsize=(14, 4), dpi=128, facecolor='w', edgecolor='k')
    # for i in range(3):
    #     plt.subplot(1, 3, i+1)
    #     plt.imshow(imgs[i*2],cmap='gray')
    # plt.show()

    # path = join(Path(__file__).parent.absolute(),"train",str(task_number))
    # imgs = get_sequence(path)

    # print(find_stars_direction(imgs))
    # cv2.namedWindow("1",cv2.WINDOW_NORMAL)
    # cv2.imshow("1", imgs[0])
    # cv2.namedWindow("2",cv2.WINDOW_NORMAL)
    # cv2.imshow("2", imgs[1])
    # cv2.waitKey()
    # cv2.destroyAllWindows()
