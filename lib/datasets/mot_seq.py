import numpy as np
import os
import torch.utils.data as data
import cv2
from lib.datasets.mot_info import get_mot_info
from lib.utils.io import read_mot_results, unzip_objs
from lib.datasets.flow_utils import readFlow

MOT_info = get_mot_info()
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



class MOTSeq(data.Dataset):
    def __init__(self, data_root, seq_name, phase, min_height, min_det_score, det_root=None,
                 label_root=None, use_flow=False):

        self.data_root = data_root # such as '/home/Qiankun/Dataset/MOT/MOT16' or '/home/Qiankun/Dataset/MOT/MOT17Det'

        # such as '/home/Qiankun/Dataset/MOT/MOT16' or
        # '/home/Qiankun/Dataset/MOT/MOT16Label' or
        # '/home/liuqk/Dataset/MOT/MOT16-det-dpm-raw/'
        self.label_root = label_root if label_root is not None else data_root
        self.seq_name = seq_name # the full name of a sequence
        self.min_height = min_height
        self.min_det_score = min_det_score
        self.det_dir = det_root # if it is None, we can load the private det file from the given directory
        self.use_flow = use_flow

        self.phase = phase
        self.split = False
        if 'split' in MOT_info:
            self.split_ratio = MOT_info['split']['ratio']
            self.split = MOT_info['split']['mode'] # split

        if self.det_dir is None:
            if 'MOT16-' in self.seq_name:
                self.detector = 'DPM'
            elif 'MOT20-' in self.seq_name:
                # MOT20 sequemces are detected by FRCNN, however the confidence score in MOT20 sequences are
                # 0 or 1, which is difference from those in MOT17. Hence, we handle MOT20 specifically.
                self.detector = 'FRCNN20'
            elif 'MOT17-' in self.seq_name:
                self.detector = self.seq_name.split('-')[-1]
            elif '2DMOT2015' in self.label_root:
                self.detector = 'ACF'
            else:
                raise ValueError('Unknown type sequence name {}'.format(self.seq_name))
        else:
            if 'POI' in self.det_dir or 'poi' in self.det_dir:
                self.detector = 'POI'
            elif 'tracktor' in self.det_dir or 'Tracktor' in self.det_dir:
                self.detector = 'TRACKTOR'
            else:
                raise ValueError('Unknown type of detection directory {}'.format(self.det_dir))

        if 'MOT17Det' in self.data_root: # this means we load the images from MOT17Det
            seq_name = '-'.join(self.seq_name.split('-')[0:2]) # remove the detector name in MOT17 to load images from MOT17Det
            seq_name = seq_name.replace('MOT16', 'MOT17') # for MOT16 sequences, we replace it with MOT17 to load images from MOT17Det
        else:
            seq_name = self.seq_name

        self.im_dir = os.path.join(self.data_root, seq_name, 'img1')
        self.flow_dir = os.path.join(self.data_root, seq_name, 'flow', 'forward')

        self.im_names = sorted([name for name in os.listdir(self.im_dir) if os.path.splitext(name)[-1] == '.jpg'])
        num_frames = len(self.im_names)
        start_frame = 1
        end_frames = num_frames
        if self.phase in ['val', 'train']:
            if self.split:
                start_frame = max(1, int(self.split_ratio[self.phase][0] * num_frames))
                end_frames = min(num_frames, int(self.split_ratio[self.phase][1] * num_frames))
        self.im_names = self.im_names[start_frame-1:end_frames + 1]

        if self.det_dir is None:
            if 'DPM' in self.seq_name or 'MOT16' in self.seq_name: # DPM detections
                # check whether it is a directory that contain raw dpm detections
                if 'MOT16-det-dpm-raw' in self.label_root:
                    if 'MOT16' in self.seq_name: # mot16 sequence
                        self.det_file = os.path.join(self.label_root, self.seq_name, 'det', 'det-dpm-raw.txt')
                    else: # mot17 sequence, such as MOT17-09-DPM
                        seq_name_tmp = '-'.join((self.seq_name.split('-')[0:2]))
                        seq_name_tmp = seq_name_tmp.replace('MOT17', 'MOT16')
                        self.det_file = os.path.join(self.label_root, seq_name_tmp, 'det', 'det-dpm-raw.txt')
                else:
                    self.det_file = os.path.join(self.label_root, self.seq_name, 'det', 'det_norm.txt')
            else:
                self.det_file = os.path.join(self.label_root, self.seq_name, 'det', 'det.txt')
        else: # we load det file from the given directory
            if self.detector == 'POI':
                self.det_file = os.path.join(self.det_dir, '{}_det.txt'.format(self.seq_name))
            elif self.detector == 'TRACKTOR':
                self.det_file = os.path.join(self.det_dir, '{}.txt'.format(self.seq_name))
       
        # import pdb
        # pdb.set_trace()
        if not os.path.exists(self.det_file):
            raise ValueError('File {} not exists!'.format(self.det_file))
       
        self.dets = read_mot_results(self.det_file, is_gt=False, is_ignore=False)

        self.gt_file = os.path.join(self.label_root, self.seq_name, 'gt', 'gt.txt')
        if os.path.isfile(self.gt_file):
            self.gts = read_mot_results(self.gt_file, is_gt=True, is_ignore=False)
        else:
            self.gts = None

    def __len__(self):
        return len(self.im_names)

    def __getitem__(self, i):
        im_name = os.path.join(self.im_dir, self.im_names[i])
        im = cv2.imread(im_name) # BGR

        # frame = i + 1
        frame = int(self.im_names[i].split('.')[0])

        dets = self.dets.get(frame, [])
        tlwhs, _, scores = unzip_objs(dets)
        scores = np.asarray(scores)

        keep = (tlwhs[:, 3] >= self.min_height) & (scores > self.min_det_score)
        tlwhs = tlwhs[keep]
        scores = scores[keep]

        if self.gts is not None:
            gts = self.gts.get(frame, [])
            gt_tlwhs, gt_ids, _ = unzip_objs(gts)
        else:
            gt_tlwhs, gt_ids = None, None

        if self.use_flow:
            flow_name = os.path.join(self.flow_dir, self.im_names[i].replace('.jpg', '.flo'))
            if os.path.exists(flow_name):
                flow = readFlow(flow_name)
            else:
                flow = None
        else:
            flow = None
        data = {
            'im': im,
            'tlwhs': tlwhs,
            'scores': scores,
            'gt_tlwhs': gt_tlwhs,
            'gt_ids': gt_ids,
            'flow': flow,
        }

        return data


def collate_fn(data):
    return data[0]


def get_seq_loader(data_root, label_root, seq_name, phase, min_height=0, min_det_score=-np.inf,
                   det_root=None, use_flow=False, num_workers=3):
    dataset = MOTSeq(data_root=data_root, label_root=label_root, seq_name=seq_name, phase=phase, min_height=min_height,
                     min_det_score=min_det_score, det_root=det_root, use_flow=use_flow)
    data_loader = data.DataLoader(dataset, 1, False, num_workers=num_workers, collate_fn=collate_fn)

    return data_loader

