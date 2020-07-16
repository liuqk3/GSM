import os
import numpy as np
# from lib.datasets.tools.misc import filter_mot_gt_boxes
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


# statics on det confidences
dataset = ['MOT16', 'MOT17']
stage = ['train', 'test']

max_num_box_per_frame = -1
xywh = np.zeros((0, 4))
for d in dataset:
    for s in stage:
        sub_d = mot[d][s]
        for seq in sub_d:
            if d in ['MOT17']:
                # there is no need to normalize the score for FRCNN, SDP
                seq = seq + '-DPM'

            # det_path = os.path.join(mot_dir, d+'Labels', s, seq, 'det', 'det.txt')
            det_path = os.path.join(mot_dir, d, s, seq, 'det', 'det.txt')
            print('normalize ', det_path)
            det = pandas.read_csv(det_path).values

            if d in ['MOT16', 'MOT17']:
                score = det[:, 6]

                min_s = score.min()

                score = score - min_s

                max_s = score.max()
                score = score / max_s

                det[:, 6] = score

            # det_norm_path = os.path.join(mot_dir, d+'Labels', s, seq, 'det', 'det_norm.txt')
            det_norm_path = os.path.join(mot_dir, d, s, seq, 'det', 'det_norm.txt')
            fmt = ['%d', '%d', '%.5f', '%.5f', '%.5f', '%.5f', '%.5f', '%.5f', '%.5f', '%.5f']
            np.savetxt(det_norm_path, det, fmt=fmt, delimiter=',')




