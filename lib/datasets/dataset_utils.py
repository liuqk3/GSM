import numpy as np
import json
from torch.utils.data import DataLoader
import torch
import cv2
from lib.utils import bbox as bbox_utils
import matplotlib.pyplot as plt

def get_neighbor(x, index):
    """Given the index, this function will pick up the corresponding
    data along the first dimension (start from zero), and return a new
    array

    Args:
        X: array like data, [N, M, *], where * denotes any number of dimensions
        index: array like data, [N, m], max(index) <= M. Each col in 'index' is the index
            of the data to pick up.
    Return:
        array like data, [N, m, *]
    """

    if isinstance(x, torch.Tensor):
        num_sample = x.size(0)
        pick_up = []
        for i in range(num_sample):
            tmp = x[i][index[i]]
            pick_up.append(tmp.unsqueeze(dim=0))
        pick_up = torch.cat(pick_up, dim=0)
        return pick_up
    elif isinstance(x, np.ndarray):
        num_sample = x.shape[0]
        pick_up = []
        for i in range(num_sample):
            tmp = x[i][index[i]]
            pick_up.append(tmp)
        pick_up = np.stack(pick_up, axis=0)
        return pick_up


def filter_mot_gt_boxes(gt_boxes, vis_threshold=0.25, min_height=2, min_width=2,
                        ambiguous_class_id=None, strict_label=True, im_shape=None,
                        year='MOT15'):
    """
    This function filter the gt boxes of mot17 and 16

    Args:
        gt_boxes: array_like, the raw data load from file gt.txt
        vis_threshold: the threshold of visibility, lower will be filtered out
        min_height: int, the boxes with the height less than this value will be filtered out
        min_width: int, the boxes with the width less than this value will be filtered out
        ambiguous_class_id: if ambiguous is None and strict_label is true, we will not
            preserve the ambiguous class, else we will preserve those boxes
        strict_label: bool, if true, we only keep Pedstrain (class 1), other wise we keep
            class 1, 2, 7
        im_shape: tuple or list, the size of image, [height, width]
    Return:
        filtered gt_boxes

    The classes of MOT17 are as follows:
        class_id = {'Pedestrain': 1,
                    'person_on_vehicle': 2,
                    'Car': 3,
                    'Bicycle': 4,
                    'Motorbike': 5,
                    'Non_motorized_vehicle': 6,
                    'Static_person': 7,
                    'Distractor': 8,
                    'Occluder': 9,
                    'Occluder_on_the_ground': 10,
                    'Occluder_full': 11,
                    'Reflection': 12}
    Each row in MOT gt file is:
    <frame id>, <ID>, <x1>, <y1>, <width>, <height>, <score/flag>, <class label>, <visibility>
    """
    assert year in ['MOT15', 'MOT16', 'MOT17', 'MOT20']

    # keep those boxes that
    idx = gt_boxes[:, 6] == 1
    gt_boxes = gt_boxes[idx]

    if year != 'MOT15':
        if strict_label:
            idx = gt_boxes[:, 7] == 1
        else:
            idx1 = gt_boxes[:, 7] == 1
            idx2 = gt_boxes[:, 7] == 2
            idx7 = gt_boxes[:, 7] == 7
            idx = idx1 + idx2 + idx7
            if ambiguous_class_id is not None:
                idx8 = gt_boxes[:, 7] == 8
                idx = idx + idx8

        gt_boxes = gt_boxes[idx]

        # the 8th (starts from 0) denotes the visibility of this object, we use vis_threshold
        # to filter those boxes have low visibility
        idx = gt_boxes[:, 8] >= vis_threshold
        gt_boxes = gt_boxes[idx]
    idx = gt_boxes[:, 4] >= min_width
    gt_boxes = gt_boxes[idx]
    idx = gt_boxes[:, 5] >= min_height
    gt_boxes = gt_boxes[idx]

    if im_shape is not None:
        height, width = im_shape[0], im_shape[1]

        # change tlwh to tlbr
        gt_boxes[:, 4] = gt_boxes[:, 2] + gt_boxes[:, 4] # x2
        gt_boxes[:, 5] = gt_boxes[:, 3] + gt_boxes[:, 5] # y2

        gt_boxes[:, 2] = np.clip(gt_boxes[:, 2], a_min=0, a_max=width - 1) # x1
        gt_boxes[:, 3] = np.clip(gt_boxes[:, 3], a_min=0, a_max=height - 1) # y1
        gt_boxes[:, 4] = np.clip(gt_boxes[:, 4], a_min=0, a_max=width - 1)  # x2
        gt_boxes[:, 5] = np.clip(gt_boxes[:, 5], a_min=0, a_max=height - 1)  # y2

        # change tlbr to tlwh
        gt_boxes[:, 4] = gt_boxes[:, 4] - gt_boxes[:, 2] # x2
        gt_boxes[:, 5] = gt_boxes[:, 5] - gt_boxes[:, 3] # y2


        idx_w = gt_boxes[:, 4] >= min_width
        idx_h = gt_boxes[:, 5] >= min_height
        idx = idx_w * idx_h
        gt_boxes = gt_boxes[idx]

    return gt_boxes


def shift_box(tlbr, shift=[0, 0]):
    """This function shit the boxes
    Args:
        tlbr: 2D array like, each row is a box denoted by [x1, y1, x2, y2]
        shift: list, [dx, dy]
    """
    if isinstance(tlbr, np.ndarray):
        tlbr[:, 0::2] = tlbr[:, 0::2] + shift[0]
        tlbr[:, 1::2] = tlbr[:, 0::2] + shift[1]

    else:
        raise NotImplementedError

    return tlbr


def flip_data(im, tlbr, type='hor'):
    """This function flip the data
    Args:
        im: 3D array like data, [h, w, c]
        tlbr: 2D array, boxes, [x1, y1, x2, y2]
        type: str, 'ver' or 'hor'
    """
    if isinstance(im, np.ndarray):
        height, width = im.shape[0], im.shape[1]
        if type == 'hor':
            im = im[:, ::-1, : ] # [h, w, c]
            new_tlbr = tlbr.copy()
            new_tlbr[:, 0] = width - 1 - tlbr[:, 2] # x1
            new_tlbr[:, 2] = width - 1 - tlbr[:, 0] # x2

        elif type == 'ver':
            im = im[::-1, :, :]  # [h, w, c]
            new_tlbr = tlbr.copy()
            new_tlbr[:, 1] = height - 1 - tlbr[:, 3]  # x1
            new_tlbr[:, 3] = height - 1 - tlbr[:, 1]  # x2
        else:
            raise ValueError('Unrecognized flip type {}!'.format(type))
    else:
        raise  NotImplementedError

    return im, new_tlbr


def flip_boxes(tlbr, im_size, type='hor'):
    """This function flip the data
    Args:
        tlbr: 2D array, boxes, [x1, y1, x2, y2]
        im_size: list or tuple, [height, width]
        type: str, 'ver' or 'hor'
    """
    if isinstance(tlbr, np.ndarray):
        new_tlbr = tlbr.copy()
    elif isinstance(tlbr, torch.Tensor):
        new_tlbr = tlbr.clone()
    else:
        raise NotImplementedError

    height, width = im_size[0], im_size[1]
    if type == 'hor':
        new_tlbr[:, 0] = width - 1 - tlbr[:, 2]  # x1
        new_tlbr[:, 2] = width - 1 - tlbr[:, 0]  # x2
    elif type == 'ver':
        new_tlbr[:, 1] = height - 1 - tlbr[:, 3]  # x1
        new_tlbr[:, 3] = height - 1 - tlbr[:, 1]  # x2
    else:
        raise ValueError('Unrecognized flip type {}!'.format(type))

    return new_tlbr



def get_adjancent_matrix(obj_ids, frame_id):
    """This function get the adjancet matrix

    Args:
        obj_ids: 1D array, [N], where N is the number of nodes. Note that the nodes
            with id set to 0 is padded nodes, and the nodes with id set to -1 is the
            empty nodes.
        frame_id: 1D array, [N], where N is the number of nodes
    """
    num_nodes = obj_ids.shape[0]
    adjacent_matrix = np.zeros((num_nodes, num_nodes))

    unique_obj_ids = list(np.unique(obj_ids))
    unique_frame_id = list(np.unique(frame_id))
    unique_frame_id.sort()

    # 1) link the nodes along tracklets
    for one_obj_id in unique_obj_ids:
        if one_obj_id == -1:
            # the empty nodes are not linked to form a tracklet.
            # Note that an object with id > 0 may disappeared in the
            # media frame, and reappear in the following frames, i.e.
            # the tracklet may be broken. The broken tracklets are now
            # not linked.
            continue
        if one_obj_id == 0:
            # the padded nodes are not linked to form a tracklet
            continue

        obj_id_idx = obj_ids == one_obj_id # [N]
        # get the node indices of this object, and arrange the indices with ascending frame id
        frame_id_tmp = frame_id[obj_id_idx]
        sort_idx = np.argsort(frame_id_tmp)
        obj_id_idx, = np.nonzero(obj_id_idx)
        obj_id_idx = obj_id_idx[sort_idx]

        for i in range(obj_id_idx.shape[0]-1):
            idx = obj_id_idx[i]
            idx_next = obj_id_idx[i+1]
            if frame_id[idx_next] == unique_frame_id[-1]:
                # the last frame is the current (or detection) frame,
                # and it does not belong to the track
                break
            else:
                adjacent_matrix[idx, idx_next] = 1
                adjacent_matrix[idx_next, idx] = 1

    # 2) link the detections with tracks
    # all detections is linked with all tracks.
    # We now only link detection nodes with the
    # track nodes in previous frame. However, we
    # can try:
    # TODO: (1) link detection nodes with all valid track nodes along the tracklet.
    # TODO: (2) link detection nodes with track nodes that in a smaller area
    det_idx = frame_id == unique_frame_id[-1]
    det_idx, = np.nonzero(det_idx)
    det_idx = list(det_idx)
    track_idx = frame_id == unique_frame_id[-2]
    track_idx, = np.nonzero(track_idx)
    track_idx = list(track_idx)

    for d_idx in det_idx:
        for t_idx in track_idx:
            if obj_ids[d_idx] == -1 or obj_ids[t_idx] == -1:
                # pass empty nodes
                continue
            # if obj_ids[d_idx] == 0 or obj_ids[t_idx] == 0:
            #     # both nodes are padded nodes, pass
            #     continue
            adjacent_matrix[d_idx, t_idx] = 1
            adjacent_matrix[t_idx, d_idx] = 1

    # 3) self link
    adjacent_matrix = adjacent_matrix + np.eye(num_nodes)

    # normalize the adjacent matrix with degree matrix
    adjacent_matrix = normalize_adjacent_matrix(adjacent_matrix=adjacent_matrix)

    return adjacent_matrix


def normalize_adjacent_matrix(adjacent_matrix):
    """Notmalize the adjacent matrix"""

    # normalize the adjacent matrix with degree matrix
    # deg_matrix = get_degree_matrix(adjacent_matrix=adjacent_matrix)
    # deg_matrix_sqrt = np.power(deg_matrix, -0.5)
    deg_diag = adjacent_matrix.sum(axis=1)  # row sum
    deg_diag_sqrt = np.power(deg_diag, -0.5)
    deg_matrix_sqrt = np.diag(deg_diag_sqrt)
    deg_matrix_sqrt[np.isinf(deg_matrix_sqrt)] = 0 # [num_nodes, num_bodes]

    adjacent_matrix = deg_matrix_sqrt.dot(adjacent_matrix).dot(deg_matrix_sqrt)
    return adjacent_matrix


def get_degree_matrix(adjacent_matrix):
    """This function computes the degree matrix

    Args:
        adjacent_matrix: 2D array-like, [N, N], where N is the number of nodes

    """
    if isinstance(adjacent_matrix, np.ndarray):
        degree = adjacent_matrix.sum(axis=1)
        degree_matrix = np.diag(degree)
    elif isinstance(adjacent_matrix, torch.Tensor):
        degree = adjacent_matrix.sum(dim=1)
        degree_matrix = torch.diag(degree)
    else:
        raise NotImplementedError('The type of adjacent marix is {}, which is not supported yet'.format(type(adjacent_matrix)))

    return degree_matrix


def get_gt_prob_matrix(obj_ids, frame_id):
    """This function get the gt prob matrix. Only compute the
     prob between detection nodes in current frame and track nodes
     in previous frame are.

    Args:
        obj_ids: 1D array, [N], where N is the number of nodes. Note that the nodes
            with id set to 0 is padded nodes, and the nodes with id set to -1 is the
            empty nodes.
        frame_id: 1D array, [N], where N is the number of nodes
    """
    unique_frame_id = list(np.unique(frame_id))
    unique_frame_id.sort()

    det_idx = frame_id == unique_frame_id[-1]
    track_idx = frame_id == unique_frame_id[-2]

    det_ids = obj_ids[det_idx]
    track_ids = obj_ids[track_idx]

    prob_matrix = track_ids[:, np.newaxis] == det_ids[np.newaxis, :]
    prob_matrix = prob_matrix.astype(np.float)

    # remove the empty nodes
    true_track = (track_ids >= 0).astype(np.float)
    prob_matrix = prob_matrix * true_track[:, np.newaxis]
    true_det = (det_ids >= 0).astype(np.float)
    prob_matrix = prob_matrix * true_det[np.newaxis, :]

    # handle the disappearance and appearance of objects
    # get the index of padded detection and track
    pad_det_idx = det_ids == 0
    pad_det_idx, = np.nonzero(pad_det_idx)
    assert pad_det_idx.shape[0] == 1
    pad_det_idx = int(pad_det_idx[0])

    pad_track_idx = track_ids == 0
    pad_track_idx, = np.nonzero(pad_track_idx)
    assert pad_track_idx.shape[0] == 1
    pad_track_idx = pad_track_idx[0]

    det_ids_list = det_ids[det_ids > 0]
    det_ids_list = list(det_ids_list)

    track_ids_list = track_ids[track_ids > 0]
    track_ids_list = list(track_ids_list)

    for r in range(track_ids.shape[0]):
        t_id = track_ids[r]
        if t_id > 0 and t_id not in det_ids_list: # disappeared
            prob_matrix[r, pad_det_idx] = 1

    for c in range(det_ids.shape[0]):
        d_id = det_ids[c]
        if d_id > 0 and d_id not in track_ids_list: # appearance
            prob_matrix[pad_track_idx, c] = 1

    return prob_matrix


def im_preprocess(image, scale, mean, var):

    image = np.asarray(image, np.float32) # [h, w, 3]

    scale = np.array(scale, dtype=np.float32).reshape(1, 1, -1) # [1, 1, 3]
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, -1) # [1, 1, 3]
    var = np.array(var, dtype=np.float32).reshape(1, 1, -1) # [1, 1, 3]

    image = (image / scale - mean) / var

    image = image.transpose((2, 0, 1)) # hwc -> chw
    return image


def extract_image_patches(image, bboxes, patch_size, scale, mean, var, channel='BGR'):
    """extract_im_patch

    Args:
        image: [w, h, 3], BGR image
        bboxes: 2D array, [N, 4], [x1, y1, x2, y2]
        patch_size: [w, h]
        scale: list with length 3, the scale to divide the image value
        mean: list with length 3, the mean value of pixel values
        var: list with length 3, the variance of pixel values
    """
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)

    if channel == 'RGB':
        image = image[:, :, ::-1] # BGR -> RGB

    bboxes = bboxes.astype(np.int)
    bboxes = bbox_utils.clip_boxes(bboxes, image.shape)
    patches = []
    for idx in range(bboxes.shape[0]):
        box = bboxes[idx]
        if box[3] > box[1] and box[2] > box[0]:
            patch = image[box[1]:box[3], box[0]:box[2]]
            patch = cv2.resize(patch, patch_size)
            #
            # plt.figure()
            # plt.imshow(patch)
            # plt.show()

            patch = im_preprocess(patch, scale, mean, var) # [c, h, w]
        else:
            patch = np.zeros((3, patch_size[1], patch_size[0]))
        patches.append(patch)

    patches = np.asarray(patches, dtype=np.float32)
    return patches



def load_mot_tracklet_loader(dataset, dataset_config,
                             batch_size, shuffle=False, num_workers=4):
    """This function load the mot dataloader for training the graph
    Args:
        dataset: the class of mot dataset
        dataset_config: a dict, the configuration for the dataset
        batch_size: int, the batch size

    """
    dataset_config = dataset_config.copy()
    dataset_config['phase'] = 'train'
    train_dataset = dataset(**dataset_config)

    dataset_config['phase'] = 'val'
    val_dataset = dataset(**dataset_config)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=shuffle, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, drop_last=True)

    return train_loader, val_loader


