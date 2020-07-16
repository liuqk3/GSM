
import os
import numpy as np
from lib.datasets.dataset_utils import filter_mot_gt_boxes
import cv2
import pandas

mot_dir = '/home/liuqk/Dataset/MOT'
mot = {
    'MOT17': {
        'train': ['MOT17-13', 'MOT17-11', 'MOT17-10', 'MOT17-09', 'MOT17-05', 'MOT17-04', 'MOT17-02'],
        'test': ['MOT17-01', 'MOT17-03', 'MOT17-06', 'MOT17-07', 'MOT17-08', 'MOT17-12', 'MOT17-14']
    },
    'MOT16': {
        'train': ['MOT16-13', 'MOT16-11', 'MOT16-10', 'MOT16-09', 'MOT16-05', 'MOT16-04', 'MOT16-02'],
        'test': ['MOT16-01', 'MOT16-03', 'MOT16-06', 'MOT16-07', 'MOT16-08', 'MOT16-12', 'MOT16-14']
    },
    '2DMOT2015': {
        'train': ['ETH-Bahnhof', 'ETH-Sunnyday', 'KITTI-13', 'KITTI-17', 'PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte'],

        'test': ['ADL-Rundle-1', 'ADL-Rundle-3', 'AVG-TownCentre', 'ETH-Crossing', 'ETH-Jelmoli', 'ETH-Linthescher',
                 'KITTI-16', 'KITTI-19', 'PETS09-S2L2', 'TUD-Crossing', 'Venice-1']
    }
}

# statics on gt boxes
dataset = ['2DMOT2015', 'MOT17']
stage = ['train']

max_num_box_per_frame = -1
xywh = np.zeros((0, 4))
for d in dataset:
    for s in stage:
        sub_d = mot[d][s]
        for seq in sub_d:
            if d in ['MOT17']:
                seq = seq + '-DPM'

            one_im_path = os.path.join(mot_dir, d, s, seq, 'img1', '000001.jpg')
            one_im = cv2.imread(one_im_path)
            im_h, im_w = one_im.shape[0], one_im.shape[1]

            gt_path = os.path.join(mot_dir, d, s, seq, 'gt', 'gt.txt')
            gt = np.loadtxt(gt_path, delimiter=',')

            if d in ['MOT17']:
                gt = filter_mot_gt_boxes(gt_boxes=gt, vis_threshold=0.1)


            for id in range(1, int(np.max(gt[:, 0])) + 1):
                index = gt[:,0] == id
                if max_num_box_per_frame < index.sum():
                    max_num_box_per_frame = index.sum()
                    print('find max number of boxes in one frame: {}'.format(max_num_box_per_frame))

            one_seq_boxes = gt[:, 2:6]
            one_seq_boxes[:, [0, 2]] = one_seq_boxes[:, [0, 2]]# / im_w
            one_seq_boxes[:, [1, 3]] = one_seq_boxes[:, [1, 3]]# / im_w

            xywh = np.concatenate((xywh, one_seq_boxes), 0)

print(xywh)



whr = np.zeros((xywh.shape[0], 3))
whr[:, 0] = xywh[:, 2] # w
whr[:, 1] = xywh[:, 3] # h
whr[:, 2] = xywh[:, 3] / xywh[:, 2] # ratio, h/w

mean_whr = np.mean(whr, axis=0)
std_whr = np.std(whr, axis=0)
min_whr = np.min(whr, axis=0)
max_whr = np.max(whr, axis=0)

print('mean [w, h, h/w ]: ', mean_whr)
print('std [w, h, h/w]: ', std_whr)
print('min [w, h, h/w ]: ', min_whr)
print('max [w, h, h/w]: ', max_whr)

