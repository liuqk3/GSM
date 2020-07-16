from lib.models.detection.frcnn_fpn import FRCNN_FPN
from lib.models.net_utils import get_model_device
import torch
import numpy as np
from torchvision.ops.boxes import clip_boxes_to_image, nms

class DetectionModel(object):

    def __init__(self, model, debug=False):
        self.model = model # the SimilarityModel
        self.model.eval()
        self.device = get_model_device(self.model)
        self.debug = debug

        self.im_scale = torch.Tensor(self.model.im_info['scale']).view(1, 1, -1).to(self.device)  # [1, 1, 3]
        self.im_mean = torch.Tensor(self.model.im_info['mean']).view(1, 1, -1).to(self.device)  # [1, 1, 3]
        self.im_var = torch.Tensor(self.model.im_info['var']).view(1, 1, -1).to(self.device)  # [1, 1, 3]

    def _process_image(self, image):
        """process the image for detection

        Args:
            image: 3D array like, BGR image
        """
        if self.model.im_info['channel'] == 'RGB':
            image = image[:, :, [2, 1, 0]]  # BGR -> RGB
        if isinstance(image, np.ndarray):
            image = image.astype(np.float32)
            image = torch.Tensor(image)  # [h, w, 3]

        if not image.is_cuda:
            image = image.to(self.device)

        image = (image / self.im_scale - self.im_mean) / self.im_var
        image = image.permute(2, 0, 1).unsqueeze(dim=0).contiguous()  # [bs, 3, h, w]
        return image

    def get_detection(self, image, tracks, det_tlwh, use_tracklet=True, public=True, det_score_thr=0.5,
                      reg_score_thr=0.5, det_nms_thr=0.3, reg_nms_thr=0.6):
        """Get detections based on tracks

        Args:
            image: 3D tensor or array, [h, w, 3]
            tracks: list of STrack
            det_tlwh: 2D array, detection box
            use_tracklet: whether to use tracklet history
            public: public detections or private detections
        """
        image = self._process_image(image)

        ################
        # do detection #
        ################
        if public:

            if isinstance(det_tlwh, np.ndarray):
                det_box = det_tlwh.copy()
                det_box = torch.Tensor(det_box)
            else:
                det_box = det_tlwh.clone()

            if det_box.nelement() > 0:
                det_box[:, 2:4] = det_box[:, 0:2] + det_box[:, 2:4] # tlbr
                det_box = det_box.to(self.device)#.unsqueeze(dim=0) # [1, n, 4]
                boxes, scores = self.model.predict_boxes(images=image, boxes=det_box)

            else:
                boxes = torch.zeros(0).to(self.device)
                scores = torch.zeros(0).to(self.device)

        else:
            boxes, scores = self.model.detect(image)

        if boxes.nelement() > 0:
            boxes = clip_boxes_to_image(boxes=boxes, size=(image.size(2), image.size(3)))
            # Filter out tracks that have too low person score
            inds = torch.gt(scores, det_score_thr).nonzero().view(-1)
            det_pos = boxes[inds]
            det_scores = scores[inds]
            # nms
            keep = nms(det_pos, det_scores, det_nms_thr)
            det_scores = det_scores[keep]
            det_pos = det_pos[keep]

        else:
            det_pos = torch.zeros(0).to(self.device)
            det_scores = torch.zeros(0).to(self.device)

        ################
        # do regression #
        ################
        if use_tracklet and len(tracks) > 0:
            track_tlbr = [t.tlbr() for t in tracks]
            track_tlbr = torch.Tensor(track_tlbr).to(self.device)
            boxes, scores = self.model.predict_boxes(images=image, boxes=track_tlbr)
            boxes = clip_boxes_to_image(boxes=boxes, size=(image.size(2), image.size(3)))

            # Filter out tracks that have too low person score
            inds = torch.gt(scores, reg_score_thr).nonzero().view(-1)
            reg_pos = boxes[inds]
            reg_scores = scores[inds]

            keep = nms(reg_pos, reg_scores, reg_nms_thr)
            reg_scores = reg_scores[keep]
            reg_pos = reg_pos[keep]

        else:
            reg_pos = torch.zeros(0).to(self.device)
            reg_scores = torch.zeros(0).to(self.device)

        ################
        # merge boxes #
        ################
        # remove some detections (problem if tracks delete each other)
        for idx in range(reg_pos.size(0)):
            nms_reg_pos = torch.cat([reg_pos[idx:idx+1, :], det_pos])
            nms_reg_scores = torch.cat([torch.tensor([2.0]).to(self.device), det_scores])
            keep = nms(nms_reg_pos, nms_reg_scores, det_nms_thr)
            keep = keep[torch.ge(keep, 1)] - 1
            det_pos = det_pos[keep]
            det_scores = det_scores[keep]

            if keep.nelement() == 0:
                break
        # concate
        poses = torch.cat((det_pos, reg_pos), dim=0)
        scores = torch.cat((det_scores, reg_scores), dim=0)

        poses = poses.to(torch.device('cpu')).numpy() # [num_box, 4]
        scores = scores.to(torch.device('cpu')).numpy() # [num_box]

        return poses, scores





