import numpy as np
import cv2
import torch

# ====================================================================== #
#                            Numpy Functions                             #
# ====================================================================== #

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    boxes = np.asarray(boxes)
    if boxes.shape[0] == 0:
        return boxes
    boxes = np.copy(boxes)
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def clip_box(bbox, im_shape):
    h, w = im_shape[:2]
    bbox = np.copy(bbox)
    bbox[0] = max(min(bbox[0], w - 1), 0)
    bbox[1] = max(min(bbox[1], h - 1), 0)
    bbox[2] = max(min(bbox[2], w - 1), 0)
    bbox[3] = max(min(bbox[3], h - 1), 0)

    return bbox


def int_box(box):
    box = np.asarray(box, dtype=np.float)
    box = np.round(box)
    return np.asarray(box, dtype=np.int)


# for display
############################
def _to_color(indx, base):
    """ return (b, r, g) tuple"""
    base2 = base * base
    b = 2 - indx / base2
    r = 2 - (indx % base2) / base
    g = 2 - (indx % base2) % base
    return b * 127, r * 127, g * 127


def get_color(indx, cls_num=1):
    if indx >= cls_num:
        return (23 * indx % 255, 47 * indx % 255, 137 * indx % 255)
    base = int(np.ceil(pow(cls_num, 1. / 3)))
    return _to_color(indx, base)


def draw_detection(im, bboxes, scores=None, cls_inds=None, cls_name=None):
    # draw image
    bboxes = np.round(bboxes).astype(np.int)
    if cls_inds is not None:
        cls_inds = cls_inds.astype(np.int)
    cls_num = len(cls_name) if cls_name is not None else 2

    imgcv = np.copy(im)
    h, w, _ = imgcv.shape
    for i, box in enumerate(bboxes):
        cls_indx = cls_inds[i] if cls_inds is not None else 1
        color = get_color(cls_indx, cls_num)

        thick = int((h + w) / 600)
        cv2.rectangle(imgcv,
                      (box[0], box[1]), (box[2], box[3]),
                      color, thick)

        if cls_indx is not None:
            score = scores[i] if scores is not None else 1
            name = cls_name[cls_indx] if cls_name is not None else str(cls_indx)
            mess = '%s: %.3f' % (name, score) if cls_inds is not None else '%.3f' % (score, )
            cv2.putText(imgcv, mess, (box[0], box[1] - 12),
                        0, 1e-3 * h, color, thick // 3)

    return imgcv


def jitter_boxes(boxes, iou_thr=0.8, up_or_low=None, region=None):
    """Jitter some boxes.
    Args:
        boxes: 2D array, [N, 4], [x1, y1, w, h]
        iou_thr: the threshold to generate random boxes
        up_or_low: str, 'up' or 'low'
        region: [x1, y1, x2, y2]， the region the cantain all boxes

    """
    jit_boxes = boxes.copy() # x1, y1, w, h
    for i in range(jit_boxes.shape[0]):
        if (jit_boxes[i] > 0).sum() > 0: # if this box if not padded zero boxes
            jit_boxes[i, 0:4] = jitter_a_box(jit_boxes[i, 0:4].copy(), iou_thr=iou_thr, up_or_low=up_or_low, region=region)
    return jit_boxes

def jitter_a_box(one_box, iou_thr=None, up_or_low=None, region=None):
    """
    This function jitter a box
    :param box: [x1, y1, w, h]
    :param iou_thr: the overlap threshold
    :param up_or_low: string, 'up' or 'low'
    :param region: [x1, y1, x2, y2], the region that contain all boxes
    :return:
    """

    # get the std
    # 1.96 is the interval of probability 95% (i.e. 0.5 * (1 + erf(1.96/sqrt(2))) = 0.975)
    std_xy = one_box[2: 4] / (2 * 1.96)
    std_wh = 10 * np.tanh(np.log10(one_box[2:4]))
    std = np.concatenate((std_xy, std_wh), axis=0)

    if up_or_low == 'up':
        jit_boxes = np.random.normal(loc=one_box, scale=std, size=(1000, 4))
        if region is not None:
            jit_boxes[:, 2:4] = jit_boxes[:, 2:4] + jit_boxes[:, 0:2] - 1
            jit_boxes[:, 0] = np.clip(jit_boxes[:, 0], a_min=region[0], a_max=region[2] - 1)
            jit_boxes[:, 1] = np.clip(jit_boxes[:, 1], a_min=region[1], a_max=region[3] - 1)
            jit_boxes[:, 2] = np.clip(jit_boxes[:, 2], a_min=region[0], a_max=region[2] - 1)
            jit_boxes[:, 3] = np.clip(jit_boxes[:, 3], a_min=region[1], a_max=region[3] - 1)
            jit_boxes[:, 2:4] = jit_boxes[:, 2:4] - jit_boxes[:, 0:2] - 1

        overlap = iou(one_box, jit_boxes)
        index = overlap >= iou_thr
        index = np.nonzero(index)[0]

    elif up_or_low == 'low':
        jit_boxes = np.random.normal(loc=one_box, scale=std, size=(1000, 4))
        if region is not None:
            jit_boxes[:, 2:4] = jit_boxes[:, 2:4] + jit_boxes[:, 0:2] - 1
            jit_boxes[:, 0] = np.clip(jit_boxes[:, 0], a_min=region[0], a_max=region[2] - 1)
            jit_boxes[:, 1] = np.clip(jit_boxes[:, 1], a_min=region[1], a_max=region[3] - 1)
            jit_boxes[:, 2] = np.clip(jit_boxes[:, 2], a_min=region[0], a_max=region[2] - 1)
            jit_boxes[:, 3] = np.clip(jit_boxes[:, 3], a_min=region[1], a_max=region[3] - 1)
            jit_boxes[:, 2:4] = jit_boxes[:, 2:4] - jit_boxes[:, 0:2] - 1

        overlap = iou(one_box, jit_boxes)
        index = (overlap <= iou_thr) & (overlap >= 0)
        index = np.nonzero(index)[0]

    else:
        raise NotImplementedError

    if index.shape[0] > 0:
        choose_index = index[np.random.choice(range(index.shape[0]))]
        choose_box = jit_boxes[choose_index]
    else:
        choose_box = one_box

    return choose_box


def iou(bbox, candidates, format='tlwh'):
    """Computer intersection over union.

    Parameters
    ----------
    bbox : ndarray, 2D [n, 4] or 1D [4]`.
    candidates : ndarray, 2D, [n, 4].

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    if len(bbox.shape) == 1:
        if format == 'tlbr':
            bbox_tl, bbox_br = bbox[:2], bbox[2:]
            candidates_tl = candidates[:, :2]
            candidates_br = candidates[:, 2:]
        elif format == 'tlwh':
            bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
            candidates_tl = candidates[:, :2]
            candidates_br = candidates[:, :2] + candidates[:, 2:]

        tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
                   np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
        br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
                   np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
        wh = np.maximum(0., br - tl)

        area_intersection = wh.prod(axis=1)
        area_bbox = bbox[2:].prod()
        area_candidates = candidates[:, 2:].prod(axis=1)
    else: # len(bbox.shape) == 2:
        raise NotImplementedError

    return area_intersection / (area_bbox + area_candidates - area_intersection)

# ====================================================================== #
#                          PyTorch Functions                             #
# ====================================================================== #
def encode_boxes(boxes, im_shape, encode=True, dim_position=64, wave_length=1000, normalize=False, quantify=-1):
    """ modified from PositionalEmbedding in:
    Args:
        boxes: [bs, num_nodes, 4] or [num_nodes, 4]
        im_shape: 2D tensor, [bs, 2] or [2], the size of image is represented as [width, height]
        encode: bool, whether to encode the box
        dim_position: int, the dimension for position embedding
        wave_length: the wave length for the position embedding
        normalize: bool, whether to normalize the embedded features
        quantify: int, if it is > 0, it will be used to quantify the position of objects

    """
    batch = boxes.dim() > 2
    if not batch:
        boxes = boxes.unsqueeze(dim=0)
        im_shape = im_shape.unsqueeze(dim=0)

    if quantify > 1:
        boxes = boxes // quantify
    # in this case, the last 2 dims of input data is num_samples and 4.
    # we compute the pairwise relative postion embedings for each box
    if boxes.dim() == 3: # [bs, num_sample, 4]
        # in this case, the boxes should be tlbr: [x1, y1, x2, y2]
        device = boxes.device

        bs, num_sample, pos_dim = boxes.size(0), boxes.size(1), boxes.size(2) # pos_dim should be 4

        x_min, y_min, x_max, y_max = torch.chunk(boxes, 4, dim=2) # each has the size [bs, num_sample, 1]

        # handle some invalid box
        x_max[x_max<x_min] = x_min[x_max<x_min]
        y_max[y_max<y_min] = y_min[y_max<y_min]

        cx_a = (x_min + x_max) * 0.5 # [bs, num_sample_a, 1]
        cy_a = (y_min + y_max) * 0.5 # [bs, num_sample_a, 1]
        w_a = (x_max - x_min) + 1. # [bs, num_sample_a, 1]
        h_a = (y_max - y_min) + 1. # [bs, num_sample_a, 1]

        cx_b = cx_a.view(bs, 1, num_sample) # [bs, 1, num_sample_b]
        cy_b = cy_a.view(bs, 1, num_sample) # [bs, 1, num_sample_b]
        w_b = w_a.view(bs, 1, num_sample) # [bs, 1, num_sample_b]
        h_b = h_a.view(bs, 1, num_sample) # [bs, 1, num_sample_b]

        delta_x = ((cx_b - cx_a) / w_a).unsqueeze(dim=-1) # [bs, num_sample_a, num_sample_b, 1]
        delta_y = ((cy_b - cy_a) / h_a).unsqueeze(dim=-1) # [bs, num_sample_a, num_sample_b, 1]
        delta_w = torch.log(w_b / w_a).unsqueeze(dim=-1)  # [bs, num_sample_a, num_sample_b, 1]
        delta_h = torch.log(h_b / h_a).unsqueeze(dim=-1)  # [bs, num_sample_a, num_sample_b, 1]

        relative_pos = torch.cat((delta_x, delta_y, delta_w, delta_h), dim=-1) # [bs, num_sample_a, num_sample_b, 4]
        # if im_shape is not None:
        im_shape = im_shape.unsqueeze(dim=-1) # [bs, 2, 1]
        im_width, im_height = torch.chunk(im_shape, 2, dim=1) # each has the size [bs, 1, 1]
        x = ((cx_b - cx_a) / im_width).unsqueeze(dim=-1) # [bs, num_sample_a, num_sample_b, 1]
        y = ((cy_b - cy_a) / im_height).unsqueeze(dim=-1) # [bs, num_sample_a, num_sample_b, 1]
        # w = ((w_b + w_a) / (2 * im_width)).unsqueeze(dim=-1) - 0.5 # [bs, num_sample_a, num_sample_b, 1]
        # h = ((h_b + h_a) / (2 * im_height)).unsqueeze(dim=-1) - 0.5 # [bs, num_sample_a. num_sample_b, 1]
        w = ((w_b - w_a) / im_width).unsqueeze(dim=-1) # [bs, num_sample_a, num_sample_b, 1]
        h = ((h_b - h_a) / im_height).unsqueeze(dim=-1) # [bs, num_sample_a. num_sample_b, 1]

        relative_pos = torch.cat((relative_pos, x, y, w, h), dim=-1) # [bs, num_sample_a, num_sample_b, 8]

        if not encode:
            embedding = relative_pos
        else:
            position_mat = relative_pos # [bs, num_sample_a, num_sample_b, 8]
            pos_dim = position_mat.size(-1)
            feat_range = torch.arange(dim_position / (2*pos_dim)).to(device) # [self.dim_position / 16]
            dim_mat = feat_range / (dim_position / (2*pos_dim))
            dim_mat = 1. / (torch.pow(wave_length, dim_mat)) # [self.dim_position / 16]

            dim_mat = dim_mat.view(1, 1, 1, 1, -1) # [1, 1, 1, 1, self.dim_position / 16]
            # position_mat = position_mat.view(bs, num_sample, num_sample, pos_dim, -1) # [bs, num_sample_a, num_sample_b, 4, 1]
            position_mat = position_mat.unsqueeze(dim=-1) # [bs, num_sample_a, num_sample_b, 8, 1]
            position_mat = 100. * position_mat # [bs, num_sample_a, num_sample_b, 8, 1]

            mul_mat = position_mat * dim_mat # [bs, num_sample_a, num_sample_b, 8, dim_position / 16]
            mul_mat = mul_mat.view(bs, num_sample, num_sample, -1) # [bs, num_sample_a, num_sample_b, dim_position / 2]
            sin_mat = torch.sin(mul_mat)# [bs, num_sample_a, num_sample_b, dim_position / 2]
            cos_mat = torch.cos(mul_mat)# [bs, num_sample_a, num_sample_b, dim_position / 2]
            embedding = torch.cat((sin_mat, cos_mat), -1)# [bs, num_sample_a, num_sample_b, dim_position]

        if normalize:
            embedding = embedding / torch.clamp(torch.norm(embedding, dim=-1, p=2, keepdim=True), 1e-6)

    else:
        raise ValueError("Invalid input of boxes.")
    if not batch: # 2D tensor, [num_boxes, 4]
        embedding = embedding.squeeze(dim=0)

    return relative_pos, embedding


def inverse_encode_boxes(boxes, relative_pos, im_shape=None, quantify=-1):
    """ This function get the anchor boxes from the boxes of neighbors and relative position:
    Args:
        boxes: [bs, neighbor_k, 4] or [neighbor_k]
        relative_pos: [bs, neighbor_k, 8] or [neighbor_k, 8]
        im_shape: 2D tensor, [bs, 2], [width, height]
        quantify: int, if it is > 0, it will be used to quantify the position of objects

    """
    batch = boxes.dim() > 2
    if not batch:
        boxes = boxes.unsqueeze(dim=0)
        relative_pos = relative_pos.unsqueeze(dim=0)
        if im_shape is not None:
            im_shape = im_shape.unsqueeze(dim=0)

    if quantify > 1:
        boxes = boxes // quantify

    # in this case, the last 2 dims of input data is num_samples and 4.
    # we try to get the anchor box based on the boxes of neighbors and relative position
    if boxes.dim() == 3:

        delta_x, delta_y, delta_w, delta_h, x, y, w, h = torch.chunk(relative_pos, 8, dim=2) # each has the size [bs, neighbor_k, 1]

        x_min, y_min, x_max, y_max = torch.chunk(boxes, 4, dim=2)
        # handle some invalid box
        x_max[x_max<x_min] = x_min[x_max<x_min]
        y_max[y_max<y_min] = y_min[y_max<y_min]

        cx_n = (x_min + x_max) * 0.5 # [bs, neighbor_k, 1]
        cy_n = (y_min + y_max) * 0.5 # [bs, neighbor_k, 1]
        w_n = (x_max - x_min) + 1. # [bs, neighbor_k, 1]
        h_n = (y_max - y_min) + 1. # [bs, neighbor_k, 1]

        # get the size of anchors based on the first 4 elements of relative position
        w_a = w_n / torch.exp(delta_w) # [bs, neighbor_k, 1]
        h_a = h_n / torch.exp(delta_h)
        cx_a = cx_n - delta_x * w_a
        cy_a = cy_n - delta_y * h_a

        x_amax = cx_a + 0.5 * w_a
        x_amin = cx_a - 0.5 * w_a
        y_amax = cy_a + 0.5 * h_a
        y_amin = cy_a - 0.5 * h_a

        box_a_1 = torch.cat((x_amin, y_amin, x_amax, y_amax), dim=-1) # [bs, neighbor_k, 4]

        if im_shape is not None:
            # get the size of anchors based on the last 4 elements of relative position
            im_shape = im_shape.unsqueeze(dim=-1) # [bs, 2, 1]
            im_width, im_height = torch.chunk(im_shape, 2, dim=1) # each has the size [bs, 1, 1]
            cx_a = cx_n - x * im_width
            cy_a = cy_n - y * im_height
            w_a = w_n - w * im_width
            h_a = h_n - h * im_height

            x_amax = cx_a + 0.5 * w_a
            x_amin = cx_a - 0.5 * w_a
            y_amax = cy_a + 0.5 * h_a
            y_amin = cy_a - 0.5 * h_a

            box_a_2 = torch.cat((x_amin, y_amin, x_amax, y_amax), dim=-1) # [bs, neighbor_k, 4]

            box_a = (box_a_1 + box_a_2) / 2
        else:
            box_a = box_a_1

    else:
        raise ValueError("Invalid input of boxes.")

    box_a = box_a * quantify
    box_a = box_a.mean(dim=1, keepdim=True) # [bs, 1, 4]

    if not batch:
        box_a = box_a.squeeze(dim=0)

    return box_a


def box_dist(tlbr):
    """This function compute the Euclidean distance between the center coordinates of boxes.
    Args:
        tlbr: 2D ([N, 4]) or 3D tensor ([bs, N, 4])
    """

    if tlbr.dim() == 2:
        num_box = tlbr.size(1)
        center = (tlbr[:, 0:2] + tlbr[:, 2:4])/2 # [num_box, 2 ]
        dist = center.view(num_box, 1, 2) - center.view(1, num_box, 2) # [num_box, num_box, 2]
        dist = dist * dist
        dist = dist.sum(dim=-1) # [num_box, num_box]
    elif tlbr.dim() == 3:
        bs, num_box = tlbr.size(0), tlbr.size(1)
        center = (tlbr[:, :, 0:2] + tlbr[:, :, 2:4]) / 2 # [bs, num_box, 2]
        dist = center.view(bs, num_box, 1, 2) - center.view(bs, 1, num_box, 2) # [bs, num_box, num_box, 2]
        dist = dist * dist
        dist = dist.sum(dim=-1) # [bs, num_box, num_box]
    else:
        raise NotImplementedError

    return dist


if __name__ == '__main__':

    # box_t = torch.Tensor([[877.81, 527.09, 905.094, 611.753], [1300.6, 455.86, 1364.5, 649.8]])
    box_d = torch.Tensor([[1310.6, 461.78, 1370.229, 642.67], [1698.6, 389.02, 1818.86, 751.79]])
    box_t = torch.Tensor([[1300.6, 441.78, 1390.229, 646.67], [1698.6, 389.02, 1818.86, 751.79]])


    im_shape = torch.Tensor([1920, 1080])

    pos_t = encode_boxes(boxes=box_t, im_shape=im_shape, encode=True)
    pos_d = encode_boxes(boxes=box_d, im_shape=im_shape, encode=True)

    relative_pos_t = pos_t[0, 1]
    relative_pos_d = pos_d[0, 1]

    diff = (relative_pos_d - relative_pos_t).abs().sum()
    a = 1

