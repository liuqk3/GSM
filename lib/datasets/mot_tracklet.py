import os
import numpy as np
import torch.utils.data as data
from torch.utils.data import DataLoader
import glob
import configparser
import torch
import pandas
import pdb
import random
import pickle
import cv2
from lib.datasets.dataset_utils import extract_image_patches, get_adjancent_matrix, get_gt_prob_matrix
from lib.datasets.dataset_utils import filter_mot_gt_boxes, flip_boxes
from lib.utils.bbox import jitter_boxes
from lib.datasets.mot_info import get_mot_info
from lib.datasets.augmentation import ImageAugmentation

MOT_info = get_mot_info()


class MOTTracklet(data.Dataset):
    def __init__(self, year='', phase='',
                 max_num_node=-1, min_num_node=-1,
                 max_num_frame=20, min_num_frame=2,
                 num_frame=-1, pad_boxes=False,
                 im_info=None, cache_dir='data_cache',
                 augment=False):
        """Get the tracjectories of an sequence.

        Args:
            year: str, can be MOT16, MOT16, MOT17, which subset to use
            phase: str, train, test or val
            max_num_node: int, the max number of box, if it is negative,
                we will load as many as possible
            min_num_node: int, the min number of box, if it is negative,
                we will load as less as posipple.
            max_num_frame: int, the max number of frames
            min_num_frame: int, the min number of frames
            num_frame: int, the number of frames, if negative, we will random choose
                it in the interval [min_num_frame, max_num_frame]
            pad_boxes: bool, whether to pad a track and a detection boxes to handle the
                disappearance and appearance of objects
            im_info: dict, condatin the image pathc info: size, mean, var, scale
            cache_dir: str, the directory to cache the data
            augment: bool, whether to augment the data
        """

        self.name = 'MOTTracklet'

        assert year in MOT_info['year']

        if phase in ['val', 'train']:
            self.im_base_dir = os.path.join(MOT_info['base_path'][year]['im'], 'train')
            self.label_base_dir = os.path.join(MOT_info['base_path'][year]['label'], 'train')
        elif phase in ['test']:
            self.im_base_dir = os.path.join(MOT_info['base_path'][year]['im'], 'test')
            self.label_base_dir = os.path.join(MOT_info['base_path'][year]['label'], 'test')
        self.year = year
        self.phase = phase

        assert self.phase in ['train', 'val']

        self.sequences = MOT_info['sequences'][self.year][self.phase]
        self.detectors = MOT_info['detectors'][self.year][self.phase]

        self.max_num_node = max_num_node
        self.min_num_node = min_num_node

        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.num_frame = num_frame
        assert self.max_num_frame > 2 * self.min_num_frame

        self.pad_boxes = pad_boxes

        self.im_augment = ImageAugmentation() if augment else None
        self.split = False
        if 'split' in MOT_info:
            self.split_ratio = MOT_info['split']['ratio']
            self.split = MOT_info['split']['mode'] # True or False

        # im_info should contain the following keys:
        # size: the size to crop image patch, [w, h]
        # scale: the value to divide the image values
        # mean: the mean of pixel values to normalize the image
        # var: the variance of pixel values to normalize the image
        # channel: RGB or BGR
        self.im_info = im_info

        self.cache_dir = cache_dir  # the directory to cache the data
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        self.tracklets = self._load_annotation()

    def _get_column_of_obj_id(self, obj_id, interval):
        min_c_idx = int(obj_id * interval)
        max_c_idx = int((obj_id + 1) * interval)

        return min_c_idx, max_c_idx

    def _load_annotation(self):
        tracklets = []
        count = 0
        for seq in self.sequences:
            for det in self.detectors:
                if det != '':
                    seq_full_name = seq + '-' + det
                else:
                    seq_full_name = seq

                # print('{}: Load annotation from sequencce {}'.format(self.phase, seq_full_name))
                seq_dir = os.path.join(self.label_base_dir, seq_full_name)

                seqinfo_path = os.path.join(seq_dir, 'seqinfo.ini')
                seqinfo = configparser.ConfigParser()
                seqinfo.read(seqinfo_path)

                im_width = int(seqinfo['Sequence']['imWidth'])
                im_height = int(seqinfo['Sequence']['imHeight'])

                # get image path, we link MOT17 and MOT16 images to MOT17Det images, since they are the same
                im_ext = seqinfo['Sequence']['imExt']

                if 'MOT17Det' in self.im_base_dir:  # this means we load the images from MOT17Det
                    seq_tmp= '-'.join(seq_full_name.split('-')[0:2])  # remove the detector name in MOT17 to load images from MOT17Det
                    seq_tmp = seq_tmp.replace('MOT16', 'MOT17')  # for MOT16 sequences, we replace it with MOT17 to load images from MOT17Det
                else:
                    seq_tmp = seq_full_name
                im_dir = os.path.join(self.im_base_dir, seq_tmp, seqinfo['Sequence']['imDir'])

                num_frames = int(seqinfo['Sequence']['seqLength'])
                if self.split:
                    start_frame = max(int(self.split_ratio[self.phase][0] * num_frames), 1)
                    end_frame = min(int(self.split_ratio[self.phase][1] * num_frames), num_frames)
                else:
                    start_frame = 1
                    end_frame = num_frames

                gt_file = os.path.join(seq_dir, 'gt', 'gt.txt')
                gt_boxes = pandas.read_csv(gt_file).values
                gt_boxes = filter_mot_gt_boxes(gt_boxes=gt_boxes, vis_threshold=0.2, min_height=2, min_width=2,
                                               strict_label=True, im_shape=[im_height, im_width], year=self.year)
                gt_boxes = gt_boxes[gt_boxes[:, 0] >= start_frame]
                gt_boxes = gt_boxes[gt_boxes[:, 0] <= end_frame]

                # get the obj id in each frame
                frame_id_to_obj_id = {}
                for fid in range(start_frame, end_frame + 1):
                    idx = gt_boxes[:, 0] == fid
                    frame_boxes = gt_boxes[idx]
                    obj_ids = np.unique(frame_boxes[:, 1])
                    obj_ids = list(obj_ids)
                    obj_ids.sort()
                    obj_ids = [int(i) for i in obj_ids]
                    frame_id_to_obj_id[fid] = obj_ids

                # change gt_boxes to trajectories
                obj_ids = gt_boxes[:, 1]
                obj_ids = list(np.unique(obj_ids))
                obj_ids.sort()
                obj_ids = [int(i) for i in obj_ids]
                interval = gt_boxes.shape[1]

                seq_trajectory = np.zeros((end_frame + 1, (max(obj_ids) + 1) * interval))  # we set the index starts from 1
                # the first start_frame rows are empty

                # set frame id and the id of none exist objects to -1
                for one_obj_id in obj_ids:
                    c_idx = interval * one_obj_id
                    seq_trajectory[:, c_idx] = np.array(list(range(end_frame + 1)))
                    seq_trajectory[:, c_idx + 1] = -1

                for one_id in obj_ids:
                    idx = gt_boxes[:, 1] == one_id
                    id_boxes = gt_boxes[idx]
                    col_range = self._get_column_of_obj_id(one_id, interval)
                    for r_idx in range(id_boxes.shape[0]):
                        one_row = id_boxes[r_idx]
                        fid = int(one_row[0])
                        seq_trajectory[fid, col_range[0]:col_range[1]] = one_row
                # get tracklets
                for fid in range(self.max_num_frame + start_frame, end_frame + 1):

                    # make sure that the last frame has at least 2 objects
                    if self.min_num_node > 0 and len(frame_id_to_obj_id[fid]) <= self.min_num_node:
                        continue
                    elif len(frame_id_to_obj_id[fid]) <= 0:
                        continue
                    im_path = os.path.join(im_dir, str(fid).zfill(6)+im_ext)
                    assert os.path.exists(im_path), im_path
                    min_fid = fid - self.max_num_frame + 1  # start from 1
                    max_fid = fid
                    tracklet_boxes = seq_trajectory[min_fid:max_fid + 1]

                    # get the frame id in this tracklet
                    fid_to_choose = []
                    for fid_t in range(min_fid, max_fid+1):
                        num_obj = len(frame_id_to_obj_id[fid_t])
                        if num_obj > 0 and num_obj > self.min_num_node:
                            fid_to_choose.append(fid_t)

                    if len(fid_to_choose) < self.num_frame:
                        continue

                    # get the objects id in one tracklet
                    inter_ids = []
                    union_ids = []

                    for one_id in obj_ids:
                        add_to_inter = True
                        add_to_union = False
                        for fid_tmp in range(min_fid, max_fid + 1):
                            if one_id not in frame_id_to_obj_id[fid_tmp]:
                                add_to_inter = False

                            if one_id in frame_id_to_obj_id[fid_tmp]:
                                add_to_union = True

                        if add_to_union:
                            union_ids.append(one_id)
                        if add_to_inter:
                            inter_ids.append(one_id)
                    count += 1
                    tracklet = {
                        'seq_name': seq_full_name,
                        'count': count,
                        'tracklet_boxes': tracklet_boxes,
                        'column_interval': interval,
                        'min_frame_id': min_fid,
                        'max_frame_id': max_fid,
                        'frame_id_to_choose': fid_to_choose,
                        'frame_id': fid,
                        'frame_id_to_obj_id': frame_id_to_obj_id,
                        'inter_ids': inter_ids,
                        'union_ids': union_ids,
                        'im_dir': im_dir,
                        'im_ext': im_ext,
                        'im_path': im_path,
                        'im_width': im_width,
                        'im_height': im_height,
                    }

                    tracklets.append(tracklet)

        return tracklets  # [:200]

    def _sample_ids(self, det_ids, num_det, track_ids, num_track):
        """sample the object ids for detections and tracks

        Args:
            det_ids: list, containing the ids of detections
            num_det: int, the number of detections
            track_ids: list, containing the ids of tracks
            num_track: int, the number of tracks

        """
        assert len(det_ids) >= num_det
        assert len(track_ids) >= num_track

        inter_ids = []
        det_left_ids = []
        track_left_ids = []

        all_ids = np.array(det_ids + track_ids)
        all_ids = list(np.unique(all_ids))

        for one_id in all_ids:
            if one_id in det_ids and one_id in track_ids:
                inter_ids.append(one_id)
            elif one_id in det_ids and one_id not in track_ids:
                det_left_ids.append(one_id)
            else:
                track_left_ids.append(one_id)

        if len(inter_ids) >= num_det and len(inter_ids) >= num_track:
            if num_det == num_track:
                # we sample the same det and track id with a large prob
                prob = random.randint(1, 100) / 100
                if prob < 0.8:  # sample the same ids
                    det_sample_ids = random.sample(inter_ids, num_det)
                    track_sample_ids = det_sample_ids
                else:  # keep 1 ids different
                    random.shuffle(inter_ids)
                    det_sample_ids = inter_ids[0:num_det - 1] + random.sample(inter_ids[num_det - 1:] + det_left_ids, 1)
                    track_sample_ids = inter_ids[0:num_track - 1] + random.sample(
                        inter_ids[num_track - 1:] + track_left_ids, 1)
            else:
                if num_det < num_track:
                    random.shuffle(inter_ids)
                    det_sample_ids = inter_ids[0:num_det]
                    track_sample_ids = det_sample_ids + random.sample(track_left_ids + inter_ids[num_det:],
                                                                      num_track - len(det_sample_ids))
                else:
                    random.shuffle(inter_ids)
                    track_sample_ids = inter_ids[0:num_track]
                    det_sample_ids = track_sample_ids + random.sample(det_left_ids + inter_ids[num_track:],
                                                                      num_det - len(track_sample_ids))
        elif len(inter_ids) >= num_det and len(inter_ids) < num_track:
            random.shuffle(inter_ids)
            det_sample_ids = inter_ids[0:num_det]
            track_sample_ids = det_sample_ids + random.sample(inter_ids[num_det:] + track_left_ids,
                                                              num_track - len(det_sample_ids))
        elif len(inter_ids) < num_det and len(inter_ids) >= num_track:
            random.shuffle(inter_ids)
            track_sample_ids = inter_ids[0:num_track]
            det_sample_ids = track_sample_ids + random.sample(inter_ids[num_track:] + det_left_ids,
                                                              num_det - len(track_sample_ids))

        else:  # if len(inter_ids) < num_det and len(inter_ids) < num_track:
            det_sample_ids = inter_ids + random.sample(det_left_ids, num_det - len(inter_ids))
            track_sample_ids = inter_ids + random.sample(track_left_ids, num_track - len(inter_ids))

        return det_sample_ids, track_sample_ids

    def _prepare_data_for_batch(self, batch_size, shuffle=True):
        """Process the tracklets, so that the data can be load for a batch.

        Note that this function should be called before starting each epoch.
        """
        print('{}: prepare data for batch...'.format(self.phase))

        if shuffle and self.phase in ['train']:
            random.shuffle(self.tracklets)

        index_to_num_frame = {}  # the tracklet length, i.e. the total number of frames (including the detion frame)
        index_to_det_frame = {}  # random choose one frame as the detection (current) frame
        index_to_previous_frame = {}  # randomly choose one frame as previous frame
        index_to_sampled_ids_det = {}  # the object ids for detections to be sampled
        index_to_sampled_ids_track = {}  # the object ids for tracks to be sampled

        # get the number of batches
        num_batch = len(self.tracklets) // batch_size
        if len(self.tracklets) % batch_size != 0:
            num_batch += 1

        for b_idx in range(num_batch):

            if b_idx % 100 == 0:
                print('{}: batch index: {}/{}'.format(self.phase, b_idx, num_batch))

            # 1) we first get the num_frame for this batch
            if self.num_frame > 0:
                num_frame = self.num_frame
            else:
                if self.phase in ['train']:
                    prob = random.randint(1, 100) / 100
                else:
                    prob = 1.0
                if prob < 0.1:
                    num_frame = random.randint(self.min_num_frame, int(self.max_num_frame / 2))
                elif prob < 0.2:
                    num_frame = random.randint(int(self.max_num_frame / 2), self.max_num_frame)
                else:
                    num_frame = self.max_num_frame

            # 2) random choose one frame as the detection frame for each sample, the previous as the existing tracks
            min_s_idx = b_idx * batch_size
            max_s_idx = min((b_idx + 1) * batch_size, len(self.tracklets))

            # 2) we set the last frame as the det frame, and randomly choose a frame as previous frame
            for s_idx in range(min_s_idx, max_s_idx):
                index_to_det_frame[s_idx] = self.tracklets[s_idx]['max_frame_id']
                index_to_num_frame[s_idx] = num_frame
                # previous_frame = random.randint(self.tracklets[s_idx]['min_frame_id'] + num_frame - 2,
                #                                 self.tracklets[s_idx]['max_frame_id'] - 1)
                frame_id_to_choose = self.tracklets[s_idx]['frame_id_to_choose'].copy()
                frame_id_to_choose.remove(index_to_det_frame[s_idx])
                previous_frame = random.choice(frame_id_to_choose)
                assert previous_frame < index_to_det_frame[s_idx]
                index_to_previous_frame[s_idx] = previous_frame

            # 3) get the object ids for detections and tracks to be sampled
            num_dets = 1e5
            num_tracks = 1e5
            for s_idx in range(min_s_idx, max_s_idx):
                tracklet_info = self.tracklets[s_idx]

                det_ids = tracklet_info['frame_id_to_obj_id'][index_to_det_frame[s_idx]]  # list
                track_ids = tracklet_info['frame_id_to_obj_id'][index_to_previous_frame[s_idx]]  # list
                if num_dets > len(det_ids):
                    num_dets = len(det_ids)
                if num_tracks > len(track_ids):
                    num_tracks = len(track_ids)

            if self.max_num_node > 0:
                num_dets = min(num_dets, self.max_num_node)
                num_tracks = min(num_tracks, self.max_num_node)

            for s_idx in range(min_s_idx, max_s_idx):
                tracklet_info = self.tracklets[s_idx]

                det_ids = tracklet_info['frame_id_to_obj_id'][index_to_det_frame[s_idx]]  # list
                track_ids = tracklet_info['frame_id_to_obj_id'][index_to_previous_frame[s_idx]]  # list

                det_ids_sample, track_ids_sample = self._sample_ids(det_ids=det_ids, num_det=num_dets,
                                                                    track_ids=track_ids, num_track=num_tracks)

                index_to_sampled_ids_det[s_idx] = det_ids_sample
                index_to_sampled_ids_track[s_idx] = track_ids_sample

        self.index_to_num_frame = index_to_num_frame
        self.index_to_det_frame = index_to_det_frame
        self.index_to_previous_frame = index_to_previous_frame
        self.index_to_sampled_ids_det = index_to_sampled_ids_det
        self.index_to_sampled_ids_track = index_to_sampled_ids_track

    def _prepare_data_for_loading(self, adj_matrix=False, gt_prob_matrix=False):
        # get the boxes, ids, adjacent matrix, degree matrix, as well as the gt cost matrix
        # for each sample, we will pad a null detection and a null track to handle the appearance
        # and disappearance of objects. The padded detection (track) will be connected with all
        # tracks (detections).

        print('{}: prepare data for loading...'.format(self.phase))

        for s_idx in range(len(self.tracklets)):

            if s_idx % 100 == 0:
                print('{}: sample index: {}/{}'.format(self.phase, s_idx, len(self.tracklets)))

            # tracklet_info = self.tracklets[s_idx]

            det_anno = []
            det_frame_id = self.index_to_det_frame[s_idx]
            det_frame_idx = det_frame_id - self.tracklets[s_idx]['min_frame_id']

            for obj_id in self.index_to_sampled_ids_det[s_idx]:
                col_range = self._get_column_of_obj_id(obj_id=obj_id, interval=self.tracklets[s_idx]['column_interval'])
                one_det_anno = self.tracklets[s_idx]['tracklet_boxes'][det_frame_idx:det_frame_idx + 1,
                               col_range[0]:col_range[1]]  # 2D array
                det_anno.append(one_det_anno)

            pad_det_anno = np.zeros((1, self.tracklets[s_idx]['column_interval']))  # append a null detection to handle disappeared tracks
            pad_det_anno[:, 0] = det_frame_id  # set the frame id for this padded detection

            # random shuffle the index of det nodes before appending the pad nodes, so that the padded
            # node will always in the last position in each frame
            det_anno = np.concatenate(det_anno, axis=0)  # 2D array
            shuffle_index = list(range(det_anno.shape[0]))
            random.shuffle(shuffle_index)
            det_anno = det_anno[shuffle_index]
            # append the pad node
            if self.pad_boxes:
                det_anno = np.concatenate([det_anno, pad_det_anno], axis=0)

            track_anno = []
            track_frame_idx_max = self.index_to_previous_frame[s_idx] - self.tracklets[s_idx]['min_frame_id']
            track_frame_idx_min = track_frame_idx_max - (self.index_to_num_frame[s_idx] - 1) + 1
            assert track_frame_idx_max >= track_frame_idx_min
            for obj_id in self.index_to_sampled_ids_track[s_idx]:
                col_range = self._get_column_of_obj_id(obj_id=obj_id, interval=self.tracklets[s_idx]['column_interval'])
                one_track_anno = self.tracklets[s_idx]['tracklet_boxes'][track_frame_idx_min:track_frame_idx_max + 1,
                                 col_range[0]:col_range[1]]  # 2D array
                track_anno.append(one_track_anno)
            # append a null track to handle new detections
            pad_track_anno = np.zeros((track_frame_idx_max - track_frame_idx_min + 1, self.tracklets[s_idx]['column_interval']))
            pad_track_anno[:, 0] = track_anno[-1][:, 0]

            # random shuffle the index of track nodes before appending the pad nodes, so that the padded
            # node will always in the last position in each frame
            track_anno = np.concatenate(track_anno, axis=0)  # 2D array
            shuffle_index = list(range(track_anno.shape[0]))
            random.shuffle(shuffle_index)
            track_anno = track_anno[shuffle_index]
            # append the pad node
            if self.pad_boxes:
                track_anno = np.concatenate([track_anno, pad_track_anno], axis=0)

            # get the data for loading
            all_anno = np.concatenate([track_anno, det_anno], axis=0)

            # 1) object ids and frame id
            frame_id = all_anno[:, 0]
            object_id = all_anno[:, 1]

            # 2) gt_boxes
            im_width, im_height = self.tracklets[s_idx]['im_width'], self.tracklets[s_idx]['im_height']
            gt_tlwh = all_anno[:, 2:6]
            jit_tlwh = jitter_boxes(boxes=gt_tlwh, iou_thr=0.75, up_or_low='up', region=[0, 0, im_width, im_height])

            load_data = {
                'frame_id': frame_id,
                'object_id': object_id,
                'gt_tlwh': gt_tlwh,
                'jit_tlwh': jit_tlwh,
                'all_anno': all_anno
            }

            # 3) get normalized adjacent matrix
            if adj_matrix:
                adjacent_matrix = get_adjancent_matrix(obj_ids=object_id, frame_id=frame_id)
                load_data['adjacent_matrix'] = adjacent_matrix
            # 4) gt prob matrix
            if gt_prob_matrix:
                prob_matrix = get_gt_prob_matrix(obj_ids=object_id, frame_id=frame_id)
                load_data['prob_matrix'] = prob_matrix
            self.tracklets[s_idx]['load_data'] = load_data

    def _get_cache_file_name(self, batch_size, epoch):
        name = self.name
        if self.pad_boxes:
            name += '_PadBoxes'
        basename = [name, self.year, self.phase, str(batch_size), str(len(self.tracklets)),
                    str(self.min_num_frame), str(self.max_num_frame), str(self.min_num_node),
                    str(self.max_num_node)]

        if self.split:
            basename.append('split_{}_{}'.format(self.split_ratio[self.phase][0], self.split_ratio[self.phase][1]))

        basename.append(str(epoch))

        basename = '_'.join(basename) + '.pkl'
        cache_path = os.path.join(self.cache_dir, basename)
        return cache_path

    def prepare_data(self, batch_size, epoch=0, shuffle=True):
        """Prepare data for training or testing
        """
        raise NotImplementedError

    def __len__(self):
        return len(self.tracklets)

    def __getitem__(self, index):
        raise NotImplementedError


class MOTTrackletStack(MOTTracklet):
    def __init__(self, year='MOT17', phase='train',
                 max_num_node=-1, min_num_node=-1,
                 max_num_frame=20, min_num_frame=2,
                 num_frame=-1, pad_boxes=False,
                 im_info=None, cache_dir='data_cache',
                 augment=False):
        """Get the tracjectories of an sequence.
        The boxes in all frames in one tracklet are stacked. i.e. suppose there are
        num_frames in this tracklet, and num_box in each frame, the final boxes has the
        shape [num_frame*num_node, 4] rather than [num_frame, num_box, 4]. By doing this,
        all boxes (nodes) are formed one graph, and we get an adjacent frame for this graph.

        Args:
            year: str, can be MOT16, MOT16, MOT17, which subset to use
            phase: str, train, test or val
            max_num_node: int, the max number of box, if it is negative,
                we will load as many as possible
            min_num_node: int, the min number of box, if it is negative,
                we will load as less as posipple.
            max_num_frame: int, the max number of frames
            min_num_frame: int, the min number of frames
            num_frame: int, the number of frames, if negative, we will random choose
                it in the interval [min_num_frame, max_num_frame]
            pad_boxes: bool, whether to pad a track and a detection boxes to handle the
                disappearance and appearance of objects
            im_info: dict, condatin the image pathc info: size, mean, var, scale
            cache_dir: str, the directory to cache the data
            augment: bool, whether to augment the images

        """
        super(MOTTrackletStack, self).__init__(year=year, phase=phase, max_num_node=max_num_node,
                                               min_num_node=min_num_node, max_num_frame=max_num_frame,
                                               min_num_frame=min_num_frame, num_frame=num_frame, pad_boxes=pad_boxes,
                                               im_info=im_info, cache_dir=cache_dir, augment=augment)
        self.name = 'MOTTrackletStack'

    def prepare_data(self, batch_size, epoch=0, shuffle=True):
        """Prepare data for training or testing
        """
        cache_path = self._get_cache_file_name(batch_size=batch_size, epoch=epoch)
        #pdb.set_trace()
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as fid:
                self.tracklets = pickle.load(fid)
            print('Tracklets loaded from {}'.format(cache_path))
        else:
            self._prepare_data_for_batch(batch_size=batch_size, shuffle=shuffle)
            self._prepare_data_for_loading(adj_matrix=True, gt_prob_matrix=True)
            with open(cache_path, 'wb') as fid:
                pickle.dump(self.tracklets, fid, pickle.HIGHEST_PROTOCOL)
                print('Tracklets wrote to {}'.format(cache_path))


    def __getitem__(self, index):
        """This function prepare one sample"""
        tracklet_info = self.tracklets[index]

        if 'load_data' not in tracklet_info.keys():
            raise RuntimeError('The {} sample has no load data'.format(index))

        load_data = tracklet_info['load_data']

        object_id = load_data['object_id'].copy()
        frame_id = load_data['frame_id'].copy()
        adjacent_matrix = load_data['adjacent_matrix'].copy() # [num_nodes, num_nodes]
        prob_matrix = load_data['prob_matrix'].copy() # [num_track, num_det], where num_nodes = num_det + num_track
        boxes = load_data['jit_tlwh'].copy()
        boxes[:, 2:4] = boxes[:, 0:2] + boxes[:, 2:4] # tlbr

        flip = random.randint(0,1)
        if flip:
            boxes = flip_boxes(tlbr=boxes, im_size=(tracklet_info['im_height'], tracklet_info['im_width']), type='hor')

        # extract patch
        im_patch = np.zeros((boxes.shape[0], 3, self.im_info['size'][1], self.im_info['size'][0])) # [num_boxes. 3, h, w]
        unique_frame_id = list(np.unique(frame_id))

        all_ims = []
        for fid in unique_frame_id:
            im_path = os.path.join(tracklet_info['im_dir'], str(int(fid)).zfill(6)+tracklet_info['im_ext'])
            im = cv2.imread(im_path) # Default is BGR
            all_ims.append(im)
            if self.im_augment is not None:
                all_ims = self.im_augment(all_ims)
        for idx in range(len(unique_frame_id)):
            im = all_ims[idx]
            if flip:
                im = im[:, ::-1, :]  # [h, w, c]

            f_idx = frame_id == unique_frame_id[idx]
            im_patch[f_idx] = extract_image_patches(image=im, bboxes=boxes[f_idx],
                                                    patch_size=tuple(self.im_info['size']), scale=self.im_info['scale'],
                                                    mean=self.im_info['mean'], var=self.im_info['var'],
                                                    channel=self.im_info['channel'])

        # get the index of detection node and track node in previous frame
        det_idx = frame_id == unique_frame_id[-1] # the largest frame id is the detection (current) frame
        det_idx = np.nonzero(det_idx)[0]
        track_idx = frame_id == unique_frame_id[-2] # the second largest frame id is the previous frame
        track_idx = np.nonzero(track_idx)[0]

        # change to tensor
        boxes = torch.Tensor(boxes) # [N, 4]
        im_patch = torch.Tensor(im_patch) # [N, 3, h, w]
        object_id = torch.Tensor(object_id) #[N]
        adjacent_matrix = torch.Tensor(adjacent_matrix) #[N, N]
        prob_matrix = torch.Tensor(prob_matrix) #[num_track, num_det]

        det_idx = torch.Tensor(det_idx) # [num_det]
        track_idx = torch.Tensor(track_idx) # [num_track]

        return im_patch, adjacent_matrix, track_idx, det_idx, boxes, object_id, prob_matrix


class MOTFramePair(MOTTracklet):
    def __init__(self, year='MOT17', phase='train',
                 max_num_node=-1, min_num_node=-1,
                 max_num_frame=20, min_num_frame=2,
                 num_frame=2, pad_boxes=False,
                 im_info=None, cache_dir='data_cache', augment=False):
        """Get a pair of frames
        Two frames are loaded. The first one is treated as the previous frame, the next one is
        treated as current frame. The number of detections and tracks may differ.

        Args:
            year: str, can be MOT16, MOT16, MOT17, which subset to use
            phase: str, train, test or val
            max_num_node: int, the max number of box, if it is negative,
                we will load as many as possible
            min_num_node: int, the min number of box, if it is negative,
                we will load as less as posipple.
            max_num_frame: int, the max number of frames
            min_num_frame: int, the min number of frames
            num_frame: int, the number of frames, if negative, we will random choose
                it in the interval [min_num_frame, max_num_frame]
            pad_boxes: bool, whether to pad a track and a detection boxes to handle the
                disappearance and appearance of objects
            im_info: dict, condatin the image pathc info: size, mean, var, scale
            cache_dir: str, the directory to cache the data
            augment: bool, whether to augment the images

        """
        super(MOTFramePair, self).__init__(year=year, phase=phase, max_num_node=max_num_node,
                                           min_num_node=min_num_node, max_num_frame=max_num_frame,
                                           min_num_frame=min_num_frame, num_frame=num_frame, pad_boxes=pad_boxes,
                                           im_info=im_info, cache_dir=cache_dir, augment=augment)
        self.name = 'MOTFramePair'
        assert self.num_frame == 2


    def prepare_data(self, batch_size, epoch=0, shuffle=True):
        """Prepare data for training or testing
        """
        cache_path = self._get_cache_file_name(batch_size=batch_size, epoch=epoch)
        #pdb.set_trace()
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as fid:
                self.tracklets = pickle.load(fid)
            print('Tracklets loaded from {}'.format(cache_path))
        else:
            self._prepare_data_for_batch(batch_size=batch_size, shuffle=shuffle)
            self._prepare_data_for_loading(adj_matrix=False, gt_prob_matrix=False)
            with open(cache_path, 'wb') as fid:
                pickle.dump(self.tracklets, fid, pickle.HIGHEST_PROTOCOL)
                print('Tracklets wrote to {}'.format(cache_path))


    def __getitem__(self, index):
        """This function prepare one sample"""
        tracklet_info = self.tracklets[index]
        im_height = tracklet_info['im_height']
        im_width = tracklet_info['im_width']

        if 'load_data' not in tracklet_info.keys():
            raise RuntimeError('The {} sample has no load data'.format(index))

        load_data = tracklet_info['load_data']

        object_id = load_data['object_id'].copy() # 1D array
        frame_id = load_data['frame_id'].copy() # 1D array

        boxes = load_data['jit_tlwh'].copy()
        # boxes = load_data['all_anno'][:, 2:6].copy()
        boxes[:, 2:4] = boxes[:, 0:2] + boxes[:, 2:4] # [N, 4], tlbr

        flip = random.randint(0,1)
        if flip:
            boxes = flip_boxes(tlbr=boxes, im_size=(tracklet_info['im_height'], tracklet_info['im_width']), type='hor')

        # extract patch
        im_patch = np.zeros((boxes.shape[0], 3, self.im_info['size'][1], self.im_info['size'][0])) # [num_boxes. 3, h, w]
        unique_frame_id = list(np.unique(frame_id))

        all_ims = []
        for fid in unique_frame_id:
            im_path = os.path.join(tracklet_info['im_dir'], str(int(fid)).zfill(6)+tracklet_info['im_ext'])
            im = cv2.imread(im_path) # Default is BGR
            all_ims.append(im)
            if self.im_augment is not None:
                all_ims = self.im_augment(all_ims)
        for idx in range(len(unique_frame_id)):
            im = all_ims[idx]
            if flip:
                im = im[:, ::-1, :]  # [h, w, c]

            f_idx = frame_id == unique_frame_id[idx]
            im_patch[f_idx] = extract_image_patches(image=im, bboxes=boxes[f_idx],
                                                    patch_size=tuple(self.im_info['size']), scale=self.im_info['scale'],
                                                    mean=self.im_info['mean'], var=self.im_info['var'],
                                                    channel=self.im_info['channel'])

        # split the im_patches, boxes, obj_ids
        unique_frame_id = list(np.unique(frame_id))
        unique_frame_id.sort()
        assert len(unique_frame_id) == 2
        det_frame_id = unique_frame_id[-1]
        track_frame_id = unique_frame_id[-2]

        det_idx = frame_id == det_frame_id
        track_idx = frame_id == track_frame_id

        det_im_patch = torch.Tensor(im_patch[det_idx]) #[num_det, 3, height, width]
        track_im_patch = torch.Tensor(im_patch[track_idx]) # [num_track, 3, height, width]

        det_boxes = torch.Tensor(boxes[det_idx]) # [num_det, 4]
        track_boxes = torch.Tensor(boxes[track_idx]) # [num_track, 4]

        det_ids = torch.Tensor(object_id[det_idx]) # [num_det]
        track_ids = torch.Tensor(object_id[track_idx]) #[num_track]

        im_shape = torch.Tensor([im_width, im_height])

        return track_im_patch, det_im_patch, track_ids, det_ids, track_boxes, det_boxes, im_shape


if __name__ == '__main__':
    
    from lib.models.config.config import get_config
    config = get_config()
    dataset_config = config['MOTFramePair']

    mot_tracklet = MOTFramePair(phase='train',
                                year=dataset_config['year'],
                                max_num_frame=dataset_config['max_num_frame'],
                                min_num_frame=dataset_config['min_num_frame'],
                                num_frame=dataset_config['num_frame'],
                                max_num_node=dataset_config['max_num_node'],
                                min_num_node=dataset_config['min_num_node'],
                                im_info=dataset_config['im_info'],
                                cache_dir=dataset_config['cache_dir'],
                                augment=dataset_config['augment'])

    dataloader = DataLoader(dataset=mot_tracklet, batch_size=4, shuffle=False, num_workers=4)

    for epoch in range(0, 30):
        print('epoch {}'.format(epoch))
        dataloader.dataset.prepare_data(batch_size=dataloader.batch_size, epoch=epoch, shuffle=True)
        # for itr, data in enumerate(dataloader):
        #     print(itr)







































