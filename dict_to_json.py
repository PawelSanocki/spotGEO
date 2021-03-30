import json
import numpy as np
from os.path import join,realpath,abspath
from pathlib import Path
import validation

class Labeled_frame:
    def __init__(self, sequence_id, frame, num_objects, object_coords):
        self.sequence_id = int(sequence_id)
        self.frame = int(frame)
        self.num_objects = int(num_objects)
        self.object_coords = object_coords

def label_frame(d: dict, sequence_id: int, frame: int) -> str:
    '''
    d - dictionary with directions as keys and lists of trajectories as values
    sequence_id - id of the sequence
    frame - number of the frame in the sequence
    returns string describing the frame
    '''
    num_objects = 0
    objects_coords = []
    for key in d.keys():
        for trajectory in d[key]:
            num_objects += 1
            trajectory_copy = list(map(int, list(trajectory[frame-1])))[1:] #[1:] first coordinate is frame number!! # frame starts from 1 [frame-1]
            objects_coords.append(trajectory_copy) 
    lf = Labeled_frame(sequence_id, frame, num_objects, objects_coords)
    label_string = json.dumps(lf.__dict__)
    return label_string

if __name__ == "__main__":
    # import trajectories2 as traj
    # task_number = 1
    # path = join(Path(__file__).parent.absolute(),"groundtruth",str(task_number))
    # imgs = traj.get_sequence(path)
    # d = traj.sequence_into_trajectories(imgs)

    # results = []
    # for j in range(5120):
    #     for i in range(5):
    #         results.append(label_frame(d, j+1, i+1))
    # with open('result.json', 'w') as outfile:
    #     outfile.write(str(results).replace("'", ""))
    # #print(str(results).replace("'", ""))
    # results = str(results).replace("'", "")
    # results = json.loads(results)
    # print(validation.validate_json(results,False))

    with open('submission.json', 'r') as f:
        results = json.load(f)
    print(validation.validate_json(results,False))
    