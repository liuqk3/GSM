import numpy as np
from collections import OrderedDict, deque
import torch
from lib.utils.bbox import iou, inverse_encode_boxes

class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3
    Replaced = 4


class BaseTrack(object):
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def end_frame(self):
        return self.frame_id

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    def mark_replaced(self):
        self.state = TrackState.Replaced


class STrack(BaseTrack):

    # def __init__(self, tlwh, score, max_n_features=100, from_det=True, im_shape=None, frame_id=None):
    def __init__(self, tlwh, score, history_len=10, from_det=True,
                 im_shape=None, frame_id=None):

        self.im_shape = im_shape # [height, width]

        # wait to be activated
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.score = score

        self.kalman_filter = None
        self.mean, self.covariance = None, None

        # classification
        self.from_det = from_det
        self.tracklet_len = 0
        self.time_by_tracking = 0

        self.frame_id = frame_id
        self.det_id = -1 # used to denote the identity of detections (NOT tracks)

        # how long the history data will be preserved
        self.history_len = history_len
        self.flow = deque([], maxlen=self.history_len)
        self.cur_flow = None
        self.feature = deque([], maxlen=self.history_len)
        self.cur_feature = None
        self.neighbor = deque([], maxlen=self.history_len) # to store the neighbors information
        self.cur_neighbor = None
        self.sample_center = None # used to sample some candidates when this track is lost.

    def self_tracking(self):
        """Perform predict the box in current from previous frame

        Args:
            type: str, 'flow' or 'kalman'
        """
        return self.tlwh()

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""

        self.kalman_filter = kalman_filter
        if self.kalman_filter is not None:
            self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.track_id = self.next_id()
        self.time_since_update = 0
        self.time_by_tracking = 0
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        """Re-activate a lost track"""
        assert self.state == TrackState.Lost

        if self.kalman_filter is not None:
            self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh()))
        else:
            self._tlwh = new_track._tlwh

        if new_track.cur_flow is not None:
            self.cur_flow = new_track.cur_flow
            self.flow.append(new_track.cur_flow)  # append the last flow of new track
        if new_track.cur_feature is not None:
            self.cur_feature = new_track.cur_feature
            self.feature.append(new_track.cur_feature)
        if new_track.cur_neighbor is not None:
            self.cur_neighbor = new_track.cur_neighbor
            self.neighbor.append(new_track.cur_neighbor)

        self.time_since_update = 0
        self.time_by_tracking = 0
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def re_find(self, tlwh, feature, neighbor, frame_id, tracked=False):
        """Refind the track

        Args:
            tlwh: 1D array, the re-find position
            tracked: bool
        """
        if self.kalman_filter is not None:
            self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xyah(tlwh))
        else:
            self._tlwh = tlwh

        if tracked:
            # self.time_since_update = 0
            # self.time_by_tracking = 0
            # self.tracklet_len = 0
            self.state = TrackState.Tracked
            # self.cur_feature = feature
            # self.cur_neighbor = neighbor
            # self.is_activated = True
            # self.frame_id = frame_id

    def predict(self):
        if self.kalman_filter is not None and len(self.flow) > 0:
            raise ValueError('Only one of Kalman filter or Flow can be used!')

        if self.kalman_filter is not None:
            mean_state = self.mean.copy()
            if self.state != TrackState.Tracked:
                mean_state[7] = 0
            self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
        elif len(self.flow) > 0 and self.flow[-1] is not None:
                flow = self.flow[-1] # [h, w, 2]
                if flow.size > 0:
                    dx = flow[:, :, 0].mean()
                    dy = flow[:, :, 1].mean()
                    self._tlwh[0] = self._tlwh[0] + dx
                    self._tlwh[1] = self._tlwh[1] + dy

        if self.time_since_update > 0:
            self.tracklet_len = 0
        self.time_since_update += 1

    def update(self, new_track, frame_id):
        """
        Update a matched track. This function should be called by a tracked in previous frame

        :type new_track: STrack
        :type frame_id: int
        :return:
        """
        self.frame_id = frame_id
        self.time_since_update = 0

        # update the information only when this track is matched with a detection
        if new_track.from_det:
            self.time_by_tracking = 0
        else:
            self.time_by_tracking += 1
        self.tracklet_len += 1
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

        if new_track.cur_flow is not None:
            self.cur_flow = new_track.cur_flow
            self.flow.append(new_track.cur_flow)  # append the last flow of new track
        if new_track.cur_feature is not None:
            self.cur_feature = new_track.cur_feature
            self.feature.append(new_track.cur_feature)
        if new_track.cur_neighbor is not None:
            self.cur_neighbor = new_track.cur_neighbor
            self.neighbor.append(new_track.cur_neighbor)

        if self.kalman_filter is not None:
            self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh()))
        else:
            self._tlwh = new_track._tlwh

    def set_det_id(self, det_id):
        """set detection id"""
        self.det_id = det_id

    def set_cur_flow(self, flow):
        """set flow for this track.
        Args:
            flow: array-like, [h, w, 2]
        """
        self.cur_flow = flow

    def set_cur_feature(self, feature):
        """set feature for this track.

        Args:
            feature: dictionary, has the keys of 'ap_feat', 'pos_feat'
        """
        self.cur_feature = feature

    def set_cur_neighbor(self, neighbor):
        """set feature for this track.

        Args:
            neighbor: dictionary, has the keys of
        """
        self.cur_neighbor = neighbor

    def set_cur_tracked_neighbors(self, match_det_track):
        """Based on the data association matched results, we can get the
        neighbors for each matched track. Note that the neighbors are those
        tracked tracks.

        Args:
            match_det_track: dict. int->int. The key is the detection id,
                        and the value is the track id. Note that the neighbors
                        is obtained based on provided detections.
        """
        neighbor_det_ids = self.cur_neighbor['det_ids']

        if neighbor_det_ids is not None:
            tracked_track_ids = []
            tracked_relative_pos = []
            for idx in range(len(neighbor_det_ids)):
                det_id = neighbor_det_ids[idx]
                if det_id in match_det_track.keys():
                    tracked_track_ids.append(match_det_track[det_id])
                    tracked_relative_pos.append(self.cur_neighbor['relative_pos'][idx])
            self.cur_neighbor['tracked_track_ids'] = tracked_track_ids
            self.cur_neighbor['tracked_relative_pos'] = tracked_relative_pos
        else:
            self.cur_neighbor['tracked_track_ids'] = None
            self.cur_neighbor['tracked_relative_pos'] = None

    def set_sample_center(self, track_dict):
        """This function compute a box for each lost tracks. So that the box can be used to sample
        some candidates when it is lost and try to find it based its neighbors.

        Args:
            track_dict: dict. track_id -> track.
        """
        # self.sample_center = self.tlwh()
        # return

        neighbor_track_ids = self.cur_neighbor['tracked_track_ids']
        if neighbor_track_ids is not None:
            relative_pos = []
            neighbot_tlbr = []
            for idx in range(len(neighbor_track_ids)):
                if neighbor_track_ids[idx] in track_dict.keys():
                    relative_pos.append(self.cur_neighbor['tracked_relative_pos'][idx])
                    neighbot_tlbr.append(track_dict[neighbor_track_ids[idx]].tlbr())

            if len(relative_pos) == 0:
                self.sample_center = self.tlwh()
            else:
                relative_pos = torch.stack(relative_pos, dim=0)
                neighbot_tlbr = torch.Tensor(neighbot_tlbr).to(relative_pos.device) # [neighbor_k, 8]
                im_height, im_width = self.im_shape[0], self.im_shape[1] # [neighbor_k, 4]
                im_shape = torch.Tensor([im_width, im_height]).to(relative_pos.device)

                self.sample_center = inverse_encode_boxes(boxes=neighbot_tlbr, relative_pos=relative_pos,
                                                          im_shape=im_shape, quantify=self.cur_neighbor['pos_quantify'])

                self.sample_center = self.sample_center.squeeze(dim=0).to(torch.device('cpu')).numpy() # [4]
                self.sample_center[2:] -= self.sample_center[:2]
        else:
            self.sample_center = self.tlwh()



    def has_neighbor(self):
        """Whether this track has neighbors"""
        if self.cur_neighbor is None:
            return False
        if self.cur_neighbor['app_feat'] is None:
            return False
        return True

    def sample_candidate(self, num_sample=64, format='tlbr', image=None):
        """This function used to sample the candidates for lost tracks"""

        tlwh = self.sample_center # self.tlwh()

        # return tlwh[np.newaxis, :]  # [1, 4]

        height, width = self.im_shape[0], self.im_shape[1]

        np.random.seed(410)
        # get the std
        # 1.96 is the interval of probability 95% (i.e. 0.5 * (1 + erf(1.96/sqrt(2))) = 0.975)
        # std_xy = tlwh[2: 4] / (2 * 1.96)
        std = np.sqrt(tlwh[2] * tlwh[3]) / (2 * 1.96)
        std_xy = np.array([std, std])

        std_wh = np.tanh(np.log10(tlwh[2:4]))
        # std_wh = np.tanh(np.log10(tlwh[2:4]))/2
        std = np.concatenate((std_xy, std_wh), axis=0)

        if (std < 0).sum() > 0:
            jit_boxes = tlwh[np.newaxis, :]  # [1, 4]
        else:
            jit_boxes = np.random.normal(loc=tlwh, scale=std, size=(2000, 4))
            jit_boxes[:, 2:4] = jit_boxes[:, 2:4] + jit_boxes[:, 0:2] - 1 # x1, y1, x2, y2
            jit_boxes[:, 0] = np.clip(jit_boxes[:, 0], a_min=0, a_max=width - 1)
            jit_boxes[:, 1] = np.clip(jit_boxes[:, 1], a_min=0, a_max=height - 1)
            jit_boxes[:, 2] = np.clip(jit_boxes[:, 2], a_min=0, a_max=width - 1)
            jit_boxes[:, 3] = np.clip(jit_boxes[:, 3], a_min=0, a_max=height - 1)

            jit_boxes[:, 2:4] = jit_boxes[:, 2:4] - jit_boxes[:, 0:2] + 1  # x1, y1, w, h
            index = (jit_boxes[:, 2] > 1) * (jit_boxes[:, 3] > 1) * (jit_boxes[:, 3]/jit_boxes[:, 2] < 3)
            if index.sum() > 0:
                jit_boxes = jit_boxes[index]
                overlap = iou(bbox=tlwh, candidates=jit_boxes, format='tlwh')
                index = overlap > 0.75
                if index.sum() > 0:
                    jit_boxes = jit_boxes[index]
                    jit_boxes = jit_boxes[0:min(jit_boxes.shape[0], num_sample)]  # tlwh
                    jit_boxes = np.concatenate((jit_boxes, tlwh[np.newaxis, :]))
                else:
                    jit_boxes = tlwh[np.newaxis, :]  # [1, 4]
            else:
                jit_boxes = tlwh[np.newaxis, :]  # [1, 4]


        if format == 'tlbr':
            jit_boxes[:, 2:4] += jit_boxes[:, 0:2]
        elif format == 'tlwh':
            pass
        else:
            raise ValueError('Unknown format of boxes {}'.format(format))

        if image is not None:
            from lib.utils.visualization import plot_detections
            import matplotlib.pyplot as plt
            if isinstance(image, torch.Tensor):
                image = image.to(torch.device('cpu')).numpy()
            image = image.astype(np.uint8)
            image = plot_detections(image=image, tlbrs=jit_boxes, scores=None, color=(0, 255, 0))
            tlbr = self.tlbr()
            tlbr = tlbr[np.newaxis, :]
            image = plot_detections(image=image, tlbrs=tlbr, scores=None, color=(255, 0, 0))
            plt.clf()
            plt.imshow(image)
            plt.pause(1)

        return jit_boxes

    def tlwh_nei(self):
        """Return the neighbor boxes"""
        if self.cur_neighbor is None:
            return None
        else:
            box = self.cur_neighbor['tlbr'] # [neighbor_k, 4]
            if box is not None:
                if isinstance(box, torch.Tensor):
                    box = box.to(torch.device('cpu')).numpy()
                box[:, 2:4] = box[:, 2:4] - box[:, 0:2] + 1
            return box

    def weight_nei(self):
        """get the neighbor weight"""
        if self.cur_neighbor is None:
            return None
        else:
            weight = self.cur_neighbor['weight'] # [neighbor_k]
            if isinstance(weight, torch.Tensor):
                weight = weight.to(torch.device('cpu')).numpy()
            return weight

    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None: # no kalman filter
            return self._tlwh.copy()
        else: # with a kalman filter
            ret = self.mean[:4].copy() # cx, cy, w/h, h
            ret[2] *= ret[3] # cx, cy, w, h
            ret[:2] -= ret[2:] / 2 # x1, y1, w, h
            return ret

    def tlwh_norm(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`. The x and y coordinates are normalized by the
                width and height of image
        """
        ret = self.tlwh()
        ret[0] /= self.im_shape[1]  # im_shape: [height, width]
        ret[1] /= self.im_shape[0]
        ret[2] /= self.im_shape[1]
        ret[3] /= self.im_shape[0]
        return ret

    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh()
        ret[2:] += ret[:2]
        return ret

    def tlbr_norm(self):
        """Get current position in bounding box format `(x1, y1, x2, y2)`.
         The x and y coordinates are normalized by the width and height of image
        """
        ret = self.tlwh()
        ret[2:] += ret[:2] # tlbr
        ret[0] /= self.im_shape[1]  # im_shape: [height, width]
        ret[1] /= self.im_shape[0]
        ret[2] /= self.im_shape[1]
        ret[3] /= self.im_shape[0]

        return ret

    def xy(self):
        """ return (center x, center y)
        """
        return self.to_xyah()[0:2]

    def xy_norm(self):
        """ normalized return (center x, center y)
        """
        center_xy = self.to_xyah()[0:2]
        center_xy[0] = center_xy[0] / self.im_shape[1]
        center_xy[1] = center_xy[1] / self.im_shape[0]

        #assert 0 <= center_xy[0] <= 1 and 0 <= center_xy[1] <= 1
        return center_xy

    def tlwh_to_xyah(self, tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh())

    def tracklet_score(self):
        """
        Please refer the paper: REAL-TIME MULTIPLE PEOPLE TRACKING WITH DEEPLY LEARNED CANDIDATE SELECTION AND PERSON RE-IDENTIFICATION
        """
        score = max(0, 1 - np.log(1 + 0.05 * self.time_by_tracking)) * (self.tracklet_len - self.time_by_tracking > 2)
        return score

    def is_lost(self):
        if self.state == TrackState.Lost:
            return True
        else:
            return False

    def is_tracked(self):
        if self.state == TrackState.Tracked:
            return True
        else:
            return False

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame())


if __name__ == '__main__':

    track = STrack(tlwh=np.array([12,45, 50, 120]), score=0.6)
    track.im_shape = [1080, 1920]
    a = track.sample_candidate()
    b = track.sample_candidate()
    track._tlwh = np.array([12,45, 60, 100])
    a = track.sample_candidate()
    b = track.sample_candidate()

    e = 1

