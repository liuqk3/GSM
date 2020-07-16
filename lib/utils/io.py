import os
from typing import Dict
import numpy as np
import pprint
import json
import glob
import cv2

def write_results(filename, results_dict: Dict, data_type: str):
    if not filename:
        return
    path = os.path.dirname(filename)
    if not os.path.exists(path):
        os.makedirs(path)

    if data_type in ('mot', 'mcmot', 'lab'):
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian -1 -1 -10 {x1} {y1} {x2} {y2} -1 -1 -1 -1000 -1000 -1000 -10 {score}\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, frame_data in results_dict.items():
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in frame_data:
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, score=1.0)
                f.write(line)
    print('Save results to {}'.format(filename))


def read_results(filename, data_type: str, is_gt=False, is_ignore=False):
    if data_type in ('mot', 'lab'):
        read_fun = read_mot_results
    else:
        raise ValueError('Unknown data type: {}'.format(data_type))

    return read_fun(filename, is_gt, is_ignore)


"""
labels={'ped', ...			% 1
'person_on_vhcl', ...	% 2
'car', ...				% 3
'bicycle', ...			% 4
'mbike', ...			% 5
'non_mot_vhcl', ...		% 6
'static_person', ...	% 7
'distractor', ...		% 8
'occluder', ...			% 9
'occluder_on_grnd', ...		%10
'occluder_full', ...		% 11
'reflection', ...		% 12
'crowd' ...			% 13
};
"""


def read_mot_results(filename, is_gt, is_ignore):
    valid_labels = {1}
    ignore_labels = {2, 7, 8, 12}
    results_dict = dict()
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            for line in f.readlines():
                linelist = line.split(',')
                if len(linelist) < 7:
                    continue
                fid = int(linelist[0])
                if fid < 1:
                    continue
                tlwh = tuple(map(float, linelist[2:6]))
                target_id = int(linelist[1])

                results_dict.setdefault(fid, list())

                if is_gt:
                    if 'MOT16-' in filename or 'MOT17-' in filename:
                        label = int(float(linelist[7]))
                        mark = int(float(linelist[6]))
                        if mark == 0 or label not in valid_labels:
                            continue
                    score = 1
                elif is_ignore:
                    if 'MOT16-' in filename or 'MOT17-' in filename:
                        label = int(float(linelist[7]))
                        vis_ratio = float(linelist[8])
                        if label not in ignore_labels and vis_ratio >= 0:
                            continue
                    else:
                        continue
                    score = 1
                else:
                    score = float(linelist[6])

                results_dict[fid].append((tlwh, target_id, score))

    return results_dict


def unzip_objs(objs):
    if len(objs) > 0:
        tlwhs, ids, scores = zip(*objs)
    else:
        tlwhs, ids, scores = [], [], []
    tlwhs = np.asarray(tlwhs, dtype=float).reshape(-1, 4)

    return tlwhs, ids, scores


def save_json(in_dict, save_path):
    json_str = json.dumps(in_dict, indent=4)
    with open(save_path, 'w') as json_file:
        json_file.write(json_str)


def my_print(obj, file=None):
    if not isinstance(obj, str):
        obj = pprint.pformat(obj)
    print(obj)
    if file is not None:
        print(obj, file=file)


def images2video(im_dir, out_video_path=None, ext='.jpg', size=None, fps=10):
    """
    Change some images to a avi video using opencv
    Args:
        im_dir: the dir contains the images
        out_video_path: the output path of the video
        ext: str, the extension of the images
        size: (h, w), the size of video
        fps: the fps of videos

    Returns:

    """
    im_list = glob.glob(os.path.join(im_dir, '*'+ext))
    im_list.sort()
    im_list = im_list[0:400]

    if out_video_path is None:
        out_video_path = os.path.join(im_dir, '..', 'video.avi')
    if size is None:
        im = cv2.imread(im_list[0])
        size = (im.shape[1], im.shape[0]) # [w, h]

    video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, size)

    for im_path in im_list:
        im = cv2.imread(im_path)
        im = cv2.resize(im, size)
        video.write(im)
    video.release()
    print('The images in {} are compressed to video {}'.format(im_dir, out_video_path))
    #cv2.destroyAllWindows()

if __name__ == '__main__':

    im_dir = '/home/liuqk/Dataset/MOT/MOT17Det/test/MOT17-03/im_track/'
    images2video(im_dir=im_dir)



    pass



