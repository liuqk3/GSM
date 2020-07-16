import numpy as np
import cv2
import torch
import torch.nn.functional as F

from lib.utils import bbox as bbox_utils
from lib.models.classification.rfcn_cls import Model as CLSModel
from lib.models.net_utils import get_model_device


def _factor_closest(num, factor, is_ceil=True):
    num = float(num) / factor
    num = np.ceil(num) if is_ceil else np.floor(num)
    return int(num) * factor


def crop_with_factor(im, dest_size, factor=32, pad_val=0, basedon='min'):
    if isinstance(im, np.ndarray):
        im_size_min, im_size_max = np.min(im.shape[0:2]), np.max(im.shape[0:2])
        im_base = {'min': im_size_min,
                   'max': im_size_max,
                   'w': im.shape[1],
                   'h': im.shape[0]}
        im_scale = float(dest_size) / im_base.get(basedon, im_size_min)

        # Scale the image.
        im = cv2.resize(im, None, fx=im_scale, fy=im_scale)

        # Compute the padded image shape. Ensure it's divisible by factor.
        h, w = im.shape[:2]
        new_h, new_w = _factor_closest(h, factor), _factor_closest(w, factor)
        new_shape = [new_h, new_w] if im.ndim < 3 else [new_h, new_w, im.shape[-1]]

        # Pad the image.
        im_padded = np.full(new_shape, fill_value=pad_val, dtype=im.dtype)
        im_padded[0:h, 0:w] = im
        shape = im.shape
    elif isinstance(im, torch.Tensor):
        im_size_min, im_size_max = min(im.size(0), im.size(1)), max(im.size(0), im.size(1))
        im_base = {'min': im_size_min,
                   'max': im_size_max,
                   'w': im.size(1),
                   'h': im.size(0)}
        im_scale = float(dest_size) / im_base.get(basedon, im_size_min)

        # Scale the image.
        resize_h = int(np.ceil(im_base['h'] * im_scale))
        resize_w = int(np.ceil(im_base['w'] * im_scale))

        im = im.permute(2, 0, 1).unsqueeze(0) # [1, 3, h, w]
        im = F.interpolate(im, size=[resize_h, resize_w], mode='bilinear', align_corners=None).int().float()  # [1, 3, h, w]
        im = im[0].permute(1, 2, 0)  # [h, w, 3]

        # Compute the padded image shape. Ensure it's divisible by factor.
        h, w = im.size(0), im.size(1)
        new_h, new_w = _factor_closest(h, factor), _factor_closest(w, factor)
        new_shape = [new_h, new_w] if im.dim() < 3 else [new_h, new_w, im.size(-1)]

        # Pad the image.
        im_padded = torch.zeros(new_shape).to(im.device) + pad_val
        im_padded[0:h, 0:w, :] = im
        shape = im.size(0), im.size(1), im.size(2)
    return im_padded, im_scale, shape


class PatchClassifier(object):
    def __init__(self, model_path=None, cuda=False):
        #self.gpu = gpu
        self.cuda = cuda
        model = CLSModel(extractor='squeezenet')
        if model_path is None:
            model_path = 'model_weight/patch_classifier_squeezenet.pth'
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model = model.eval()
        if cuda:
            model = model.cuda()
        self.model = model
        print('load cls model from: {}'.format(model_path))
        self.score_map = None
        self.im_scale = 1.
        self.device = get_model_device(self.model)

    @staticmethod
    def im_preprocess(image):

        # resize and padding
        # real_inp_size = min_size
        if isinstance(image, np.ndarray):
            ori_shape = image.shape[0:2]
        elif isinstance(image, torch.Tensor):
            ori_shape = (image.size(0), image.size(1))
        if min(ori_shape) > 720:
            real_inp_size = 640
        else:
            real_inp_size = 368
        im_pad, im_scale, real_shape = crop_with_factor(image, real_inp_size, factor=16, pad_val=0, basedon='min')

        # preprocess image
        if isinstance(im_pad, np.ndarray):
            im_croped = cv2.cvtColor(im_pad, cv2.COLOR_BGR2RGB)
            im_croped = im_croped.astype(np.float32) / 255. - 0.5

        elif isinstance(im_pad, torch.Tensor):
            im_croped = im_pad[:, :, [2, 1, 0]]  # BGR -> RGB
            im_croped = im_croped / 255. - 0.5

        return im_croped, im_pad, real_shape, im_scale

    def update(self, image):

        if isinstance(image, np.ndarray):
            self.ori_image_shape = image.shape
        elif isinstance(image, torch.Tensor):
            self.ori_image_shape = image.size()

        im_croped, im_pad, real_shape, im_scale = self.im_preprocess(image)

        self.im_scale = im_scale

        if isinstance(im_croped, np.ndarray):
            im_data = torch.from_numpy(im_croped).to(self.device)
        else:
            im_data = im_croped
        im_data = im_data.permute(2, 0, 1)
        im_data = im_data.unsqueeze(0)

        with torch.no_grad():
            # device = get_model_device(self.model)
            # im_var = im_data.to(device)
            # self.score_map = self.model(im_var)
            self.score_map = self.model(im_data)

        return real_shape, im_scale

    def predict(self, rois):
        """
        :param rois: numpy array [N, 4] ( x1, y1, x2, y2)
        :return: scores [N]
        """
        scaled_rois = rois * self.im_scale
        cls_scores = self.model.get_cls_score_numpy(self.score_map, scaled_rois, self.cuda)

        # check area
        rois = rois.reshape(-1, 4)
        clipped_boxes = bbox_utils.clip_boxes(rois, self.ori_image_shape)

        ori_areas = (rois[:, 2] - rois[:, 0]) * (rois[:, 3] - rois[:, 1])
        areas = (clipped_boxes[:, 2] - clipped_boxes[:, 0]) * (clipped_boxes[:, 3] - clipped_boxes[:, 1])
        ratios = areas / np.clip(ori_areas, a_min=1e-4, a_max=None)
        cls_scores[ratios < 0.5] = 0

        return cls_scores
