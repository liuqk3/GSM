import numpy as np
import itertools
from lib.utils.nms_wrapper import nms_detections

from lib.tracking import matching
from lib.utils.kalman_filter import KalmanFilter
from .track import TrackState, STrack
from lib.datasets.flow_utils import crop_flows
import time


class Threshold(object):
    cls_score = {
        'CLASSIFIER': 0.4,
        'POI': 0.35,
        'TRACKTOR': 0.5,
        'DPM': 0.1,
        'FRCNN': 0.8,
        'SDP': 0.7,
        'FRCNN20': -1, # the score of detections in MOT20 are 0 or 1, so we set the thresehold to 0
    }
    nms = {
        'CLASSIFIER': 0.3,
        'POI': 0.45,
        'TRACKTOR': 0.3,
        'DPM': 0.3,
        'FRCNN': 0.5,
        'SDP': 0.6,
        'FRCNN20': -1,
    }
    activate = {
        'CLASSIFIER': 0.6,
        'POI': 0.5,
        'TRACKTOR': 0.8,
        'DPM': 0.2,
        'FRCNN': 0.95,
        'SDP': 0.99,
        'FRCNN20': -1,
    }


class OnlineTracker(object):

    def __init__(self, detector_name='',
                 # min_cls_score=0.4,
                 min_asso_dist=0.0015,
                 max_time_lost=30,
                 association_model=None,
                 classifier=None,
                 detection_model=None,
                 match_type='graph_match',
                 use_neighbor=True,
                 use_tracking=False,
                 use_reactive=True,
                 use_refind=False,
                 use_flow=False,
                 use_kalman_filter=False,
                 use_iou_match=True,
                 gate_cost_matrix=False,
                 strict_match=False,
                 debug=False):

        # self.min_cls_score = min_cls_score
        self.max_time_lost = max_time_lost
        self.match_type = match_type
        self.use_neighbor = use_neighbor # whether to use
        self.use_reactive = use_reactive
        self.use_tracking = use_tracking
        self.use_flow = use_flow
        self.use_iou_match = use_iou_match
        self.use_refind = use_refind

        self.strict_match = strict_match
        self.gate_cost_matrix = gate_cost_matrix
        self.debug = debug
        self.kalman_filter = KalmanFilter() if use_kalman_filter else None

        self.tracked_tracks = []   # type, list[STrack]
        self.lost_tracks = []      # type, list[STrack]
        self.removed_tracks = []   # type, list[STrack]

        self.min_asso_dist = min_asso_dist

        # similar model
        self.asso_model = association_model
        self.asso_model.debug = self.debug

        # classifier model
        self.classifier = classifier # PatchClassifier()
        self.detector = detection_model
        self.detector_name = detector_name
        if self.classifier is not None and self.detector is not None:
            raise ValueError('Only one of detector and classifier can be used!')

        if self.classifier is not None:
            self.detector_name = 'CLASSIFIER'
        if self.detector is not None:
            self.detector_name = 'TRACKTOR'

        self.frame_id = 0
        self.det_count = 0
        self.tracked_dict = {} # dictionary to preserve tracked tracks, map from  track_id -> track
        self.match_det_track = {} # map from det_id -> track_id. det_id: track_id
        self.match_track_det = {}  # map from track_id -> det_id. track_id: det_id

    def update(self, image, tlwhs, det_scores=None, flow=None):
        self.det_count = 0 # used to identify the identities of detections
        self.tracked_dict = {}
        self.match_det_track = {}
        self.match_track_det = {}

        self.frame_id += 1
        if isinstance(image, np.ndarray):
            im_shape = [image.shape[0], image.shape[1]] # [height, width]
        else: # tensor
            im_shape = [image.size(0), image.size(1)]  # [height, width]

        # filter the invalid boxes
        tlwhs[tlwhs < 0] = 0
        tlbrs = tlwhs.copy()
        tlbrs[:, 2] = tlbrs[:, 0] + tlbrs[:, 2]
        tlbrs[:, 3] = tlbrs[:, 1] + tlbrs[:, 3]
        tlbrs =tlbrs.astype(np.int)
        idx = (tlbrs[:, 0] < tlbrs[:, 2]) * (tlbrs[:, 1] < tlbrs[:, 3]) * \
              (tlbrs[:, 0] < im_shape[1]) * (tlbrs[:, 1] < im_shape[0]) * \
              (tlbrs[:, 2] > 0) * (tlbrs[:, 3] > 0)
        tlwhs = tlwhs[idx]
        if det_scores is not None:
            det_scores = det_scores[idx]

        # start to tracking
        activated_tracks = []
        reactived_tracks = []
        lost_tracks = []
        refind_tracks = []
        removed_tracks = []

        """step 1: prediction"""
        for strack in itertools.chain(self.tracked_tracks, self.lost_tracks):
            strack.predict()

            # strack.sample_candidate(image=image)

        """step 2: prepare detection"""
        if self.classifier is None and self.detector is None:
            detections = []
            for tlwh, score in zip(tlwhs, det_scores):
                if score > Threshold.cls_score[self.detector_name]:  # self.min_cls_score:
                    det = STrack(tlwh=tlwh, score=score, from_det=True, im_shape=im_shape, frame_id=self.frame_id)
                    detections.append(det)

            if self.use_tracking:
                print('Class Tracker.update(): use_tracking is True, but no classifier provided!')
                tracks = []
                for t in itertools.chain(self.tracked_tracks, self.lost_tracks):
                    if t.is_activated:
                        tracks.append(STrack(tlwh=t.self_tracking(), score=t.tracklet_score()*t.score, from_det=False))
                detections.extend(tracks)
            rois = np.asarray([d.tlbr() for d in detections], dtype=np.float32)
            scores = np.asarray([d.score for d in detections], dtype=np.float)
        elif self.classifier is not None:

            det_scores = np.ones(len(tlwhs), dtype=float)
            detections = []
            for tlwh , score in zip(tlwhs, det_scores):
                det = STrack(tlwh=tlwh, score=score, from_det=True, im_shape=im_shape, frame_id=self.frame_id)
                detections.append(det)

            self.classifier.update(image)
            n_dets = len(tlwhs)
            if self.use_tracking:
                tracks = []
                for t in itertools.chain(self.tracked_tracks, self.lost_tracks):
                    if t.is_activated:
                        tracks.append(STrack(tlwh=t.self_tracking(), score=t.tracklet_score(), from_det=False))
                detections.extend(tracks)
            else:
                print('Class Tracker.update(): use_tracking is False, but a classifier provided!')
            rois = np.asarray([d.tlbr() for d in detections], dtype=np.float32)
            scores = np.asarray([d.score for d in detections], dtype=np.float)

            cls_scores = self.classifier.predict(rois)
            scores[0:n_dets] = 1.
            scores = scores * cls_scores
        elif self.detector is not None:
            # TODO: pay attention to the detection
            rois, scores = self.detector.get_detection(image=image, tracks=self.tracked_tracks,
                                                       det_tlwh=tlwhs, use_tracklet=True, public=True)

            index = scores >= Threshold.cls_score[self.detector_name]
            rois = rois[index]
            scores = scores[index]

            if rois.size == 0:
                print('Warning: frame {} has no detection! '.format(self.frame_id))
                detections = []
            else:
                rois_tlwh = rois.copy()
                rois_tlwh[:, 2:4] = rois[:, 2:4] - rois[:, 0:2]
                detections = []
                for i in range(rois_tlwh.shape[0]):
                    det = STrack(tlwh=rois_tlwh[i], score=scores[i], from_det=True, im_shape=im_shape, frame_id=self.frame_id)
                    detections.append(det)

        # nms for non detections
        if self.detector_name != 'FRCNN20' and self.detector is None and len(detections) > 0:
            # MOT20 sequences are detected by a FRCNN, however, the confidence score in MOT20 sequences are 0 or 1
            # so we do not perform NMS on MOT20 sequences
            keep = nms_detections(rois, scores.reshape(-1), nms_thresh=Threshold.nms[self.detector_name])
            mask = np.zeros(len(rois), dtype=np.bool)
            mask[keep] = True
            keep = np.where(mask & (scores >= Threshold.cls_score[self.detector_name]))[0] # self.min_cls_score))[0]
            detections = [detections[i] for i in keep]
            scores = scores[keep]
            for d, score in zip(detections, scores):
                d.score = score

        for d in detections:
            self.det_count += 1
            d.set_det_id(self.det_count)

        if self.use_flow and flow is not None:
            tlbrs = np.array([d.tlbr() for d in detections])
            flows = crop_flows(tlbrs=tlbrs, flow=flow)
            for idx in range(len(detections)):
                detections[idx].set_cur_flow(flows[idx])

        # set feature and neighbors for detections
        self.asso_model.update_image(image=image, frame_id=self.frame_id) # set the image

        self.asso_model.set_feat_and_neighbors(tracks=detections, tracks_type='det', match_type=self.match_type,
                                               strict=self.strict_match, use_neighbor=self.use_neighbor)

        track_dets = [d for d in detections if not d.from_det] # the detections obtained based on object-self tracking
        new_dets = [d for d in detections if d.from_det]

        """step 3: data association """
        # matching for tracked targets
        unconfirmed_tracks = []
        confirmed_tracks = []  # type, list[STrack]
        for track in self.tracked_tracks:
            if not track.is_activated:
                unconfirmed_tracks.append(track)
            else:
                confirmed_tracks.append(track)

        # matching confirmed tracks with detection
        matches, u_confirmed, u_detection = self.asso_model.association(tracks=confirmed_tracks,
                                                                        detections=new_dets,
                                                                        match_type=self.match_type,
                                                                        threshold=self.min_asso_dist,
                                                                        kalman_filter=self.kalman_filter,
                                                                        gate=self.gate_cost_matrix,
                                                                        use_neighbor=self.use_neighbor)

        for itracked, idet in matches:
            confirmed_tracks[itracked].update(new_dets[idet], self.frame_id)
            self.match_det_track[new_dets[idet].det_id] = confirmed_tracks[itracked].track_id
            self.match_track_det[confirmed_tracks[itracked].track_id] = new_dets[idet].det_id
        # matching the lost tracks with unmatched detections
        new_dets = [new_dets[i] for i in u_detection]
        matches, u_lost, u_detection = self.asso_model.association(tracks=self.lost_tracks,
                                                                   detections=new_dets,
                                                                   match_type=self.match_type,
                                                                   threshold=self.min_asso_dist,
                                                                   kalman_filter=self.kalman_filter,
                                                                   gate=self.gate_cost_matrix,
                                                                   use_neighbor=self.use_neighbor)
        for ilost, idet in matches:
            track = self.lost_tracks[ilost]
            det = new_dets[idet]
            track.re_activate(det, self.frame_id, new_id=not self.use_reactive)
            reactived_tracks.append(track)
            self.match_det_track[new_dets[idet].det_id] = self.lost_tracks[ilost].track_id
            self.match_track_det[self.lost_tracks[ilost].track_id] = new_dets[idet].det_id

        # matching the unmatched confirmed tracks with remaining unmatched detection based on IoU
        if self.use_iou_match:
            len_det = len(u_detection)
            dets = [new_dets[i] for i in u_detection] + track_dets
            r_confirmed_tracks = [confirmed_tracks[i] for i in u_confirmed]

            dists = matching.iou_distance(r_confirmed_tracks, dets)
            matches, u_confirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)

            for itracked, idet in matches:
                r_confirmed_tracks[itracked].update(dets[idet], self.frame_id)
                self.match_det_track[dets[idet].det_id] = r_confirmed_tracks[itracked].track_id
                self.match_track_det[r_confirmed_tracks[itracked].track_id] = dets[idet].det_id
            for it in u_confirmed:
                track = r_confirmed_tracks[it]

                track.mark_lost()
                lost_tracks.append(track)

            new_dets = [dets[i] for i in u_detection if i < len_det]
        else:

            r_confirmed_tracks = [confirmed_tracks[i] for i in u_confirmed]
            matches, u_confirmed, u_track_detection = self.asso_model.association(tracks=r_confirmed_tracks,
                                                                                 detections=track_dets,
                                                                                 match_type=self.match_type,
                                                                                 threshold=self.min_asso_dist,
                                                                                 kalman_filter=self.kalman_filter,
                                                                                 gate=self.gate_cost_matrix,
                                                                                 use_neighbor=self.use_neighbor)

            for itracked, idet in matches:
                r_confirmed_tracks[itracked].update(track_dets[idet], self.frame_id)
                self.match_det_track[track_dets[idet].det_id] = r_confirmed_tracks[itracked].track_id
                self.match_track_det[r_confirmed_tracks[itracked].track_id] = track_dets[idet].det_id
            for it in u_confirmed:
                track = r_confirmed_tracks[it]
                track.mark_lost()
                lost_tracks.append(track)

            new_dets = [new_dets[i] for i in u_detection]

        # matching the unconfirmed_tracks tracks with remaining unmatched detection based on IoU
        if self.use_iou_match:
            dists = matching.iou_distance(unconfirmed_tracks, new_dets)
            matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        else:
            matches, u_unconfirmed, u_detection = self.asso_model.association(tracks=unconfirmed_tracks,
                                                                              detections=new_dets,
                                                                              match_type=self.match_type,
                                                                              threshold=self.min_asso_dist,
                                                                              kalman_filter=self.kalman_filter,
                                                                              gate=self.gate_cost_matrix,
                                                                              use_neighbor=self.use_neighbor) # the estimation of unconfirmed tracks made by KF may be wrong
        for itracked, idet in matches:
            unconfirmed_tracks[itracked].update(new_dets[idet], self.frame_id)
            self.match_det_track[new_dets[idet].det_id] = unconfirmed_tracks[itracked].track_id
            self.match_track_det[unconfirmed_tracks[itracked].track_id] = new_dets[idet].det_id
        for it in u_unconfirmed:
            track = unconfirmed_tracks[it]

            track.mark_removed()
            removed_tracks.append(track)

        """step 4: init new tracks"""
        for inew in u_detection:
            track = new_dets[inew]
            if not track.from_det or track.score < Threshold.activate[self.detector_name]:
                continue
            track.activate(kalman_filter=self.kalman_filter, frame_id=self.frame_id)
            activated_tracks.append(track)

            # the new inited tracks should be also be treated as matched with itself
            self.match_det_track[track.det_id] = track.track_id
            self.match_track_det[track.track_id] = track.det_id

        """step 5: update state"""
        for track in self.lost_tracks:
            if self.frame_id - track.end_frame() > self.max_time_lost:
                track.mark_removed()
                removed_tracks.append(track)

        """step 6: refind lost tracks """
        self.removed_tracks.extend(removed_tracks)
        tracked_tracks = [t for t in self.tracked_tracks if t.state == TrackState.Tracked]
        tracked_tracks.extend(activated_tracks)
        tracked_tracks.extend(reactived_tracks)
        lost_tracks.extend([t for t in self.lost_tracks if t.state == TrackState.Lost])

        # refind the lost tracks
        if self.use_refind:

            # update the neighbors of tracks, only need to update the matched tracks
            for strack in tracked_tracks:  # those lost tracks must be unmatched with any detection
                self.tracked_dict[strack.track_id] = strack
                if strack.track_id in self.match_track_det.keys():
                    strack.set_cur_tracked_neighbors(match_det_track=self.match_det_track)

            # prepare the center box for each lost track to sample some candidates
            for strack in lost_tracks:
                strack.set_sample_center(self.tracked_dict)

            support_tracks = [t for t in self.tracked_tracks if t.is_activated]
            self.asso_model.find_lost_tracks(frame_id=self.frame_id, lost_tracks=lost_tracks,
                                             support_tracks=support_tracks, match_type=self.match_type,
                                             strict=self.strict_match, use_neighbor=self.use_neighbor,
                                             threshold=self.min_asso_dist* 1)# 0.9)

            refind_tracks = [t for t in lost_tracks if t.state == TrackState.Tracked]
            tracked_tracks.extend(refind_tracks)
            lost_tracks = [t for t in lost_tracks if t.state == TrackState.Lost]

        self.tracked_tracks = tracked_tracks
        self.lost_tracks = lost_tracks

        """ 7 get the scores of lost tracks """
        # get scores of lost tracks
        if self.classifier is not None:
            rois = np.asanyarray([t.tlbr() for t in self.lost_tracks], dtype=np.float32)
            lost_cls_scores = self.classifier.predict(rois)
            out_lost_stracks = [t for i, t in enumerate(self.lost_tracks) if
                                lost_cls_scores[i] > 0.75 * Threshold.cls_score[self.detector_name] and self.frame_id - t.end_frame() <= 4]
        else:
            out_lost_stracks = [t for i, t in enumerate(self.lost_tracks) if
                                t.score > 0.75 * Threshold.cls_score[self.detector_name] and self.frame_id - t.end_frame() <= 4]
            # out_lost_stracks = []
        output_tracked_tracks = [track for track in self.tracked_tracks if track.is_activated]
        output_tracks = output_tracked_tracks + out_lost_stracks

        if self.debug:
            print('===========Frame {}=========='.format(self.frame_id))
            print('Activated: {}'.format([track.track_id for track in activated_tracks]))
            print('Reactivated: {}'.format([track.track_id for track in reactived_tracks]))
            print('Lost: {}'.format([track.track_id for track in lost_tracks]))
            print('Removed: {}'.format([track.track_id for track in removed_tracks]))
            print('Refind: {}'.format(track.track_id for track in refind_tracks))

        return output_tracks


