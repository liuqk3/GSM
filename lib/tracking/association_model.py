import torch
import pdb
import time
import itertools
import numpy as np
from lib.datasets.dataset_utils import extract_image_patches
from lib.models.net_utils import get_model_device
from lib.tracking.matching import indices_to_matches, gate_cost_matrix
from lib.utils import visualization as vis
import matplotlib.pyplot as plt
import cv2
from collections import deque
from roi_align.roi_align import RoIAlign
from roi_align.crop_and_resize import CropAndResizeFunction


class AssociationModel(object):
    def __init__(self, model, debug=False):
        self.model = model # the SimilarityModel
        self.model.eval()
        self.device = get_model_device(self.model)
        self.debug = debug

        self.score_time = 0 # used to get the time consumption for getting score
        self.construct_time = 0 # used to get  the time consumption for constructing the graph

        crop_w, crop_h = self.model.backbone.im_info['size']
        self.roi_align = RoIAlign(crop_width=crop_w, crop_height=crop_h)
        self.im_scale = torch.Tensor(self.model.backbone.im_info['scale']).view(1, 1, -1).to(self.device)  # [1, 1, 3]
        self.im_mean = torch.Tensor(self.model.backbone.im_info['mean']).view(1, 1, -1).to(self.device)  # [1, 1, 3]
        self.im_var = torch.Tensor(self.model.backbone.im_info['var']).view(1, 1, -1).to(self.device)  # [1, 1, 3]

        self.frame_id = -1
        self.image = None

        # for debug:
        self.pre_frame_id = -1
        self.det_frames = deque([], maxlen=6)

    def update_image(self, image, frame_id):
        """set current image
          image: 3D array, [height, width, 3]
        """
        self.frame_id = frame_id

        if self.model.backbone.im_info['channel'] == 'RGB':
            image = image[:, :, [2, 1, 0]] # BGR -> RGB
        if isinstance(image, np.ndarray):
            image = image.copy().astype(np.float32)
            image = torch.Tensor(image)  # [h, w, 3]

        image = image.to(self.device)

        image = (image / self.im_scale - self.im_mean) / self.im_var

        self.image = image.permute(2, 0, 1).unsqueeze(dim=0).contiguous()  # [bs, 3, h, w]

    def image2plot(self, channel='RGB'):
        """set current image
          image: 3D array, [height, width, 3]
        """

        image = self.image[0].permute(1, 2, 0)
        image = (image * self.im_var + self.im_mean) * self.im_scale
        image = image.to(torch.device('cpu')).numpy()
        image = image.astype(np.uint8)

        if self.model.backbone.im_info['channel'] != channel:
            image = image[:, :, ::-1]
        return image

    def _crop_im_patch(self, tlbr):
        """Crop image patches

        tlbr: 2D tensor, [num_box, 4]
        image: 3D array, [height, width, 3]
        """

        image = self.image

        if isinstance(tlbr, np.ndarray):
            box = torch.Tensor(tlbr).to(self.device) # 2D
        else:
            box = tlbr

        idx = torch.Tensor([0]).repeat(box.size(0)).to(self.device).int() # 1D tensor

        patches = self.roi_align(image, box, idx) # [n, 3, ch, cw]

        # patches = (patches / self.im_scale - self.im_mean) / self.im_var
        # one_p = patches[0].permute(1,2,0)
        # one_p = (one_p * self.im_var + self.im_mean) * self.im_scale
        # one_p = one_p.cpu().numpy().astype(np.uint8)
        # plt.clf()
        # plt.imshow(one_p)
        # plt.pause(0.01)

        return patches

    def set_feat_and_neighbors(self, tracks, tracks_type='det', match_type='graph_match', strict=True, use_neighbor=True):
        """This function get the neighbors for detections. Once a detection is matched with a
        track, the information of neighbors will be delivered to tracks.

        Args:
            tracks: tracks: list of STrack, it should be detections
            match_type: str, which match is used
            strict: bool, if true, we will perform association between true tracks and true detections.
                If False, we will do association between all (including padded)tracks and all (including
                padded) detections.
            use_neighbor: bool whether to use neighbors when perform graph match. Only effective in 'graph_match'
            im_shape: [width, height]
        """
        num_node = len(tracks)
        if num_node <= 0:
            return

        tlbr = [t.tlbr() for t in tracks]
        # pad box
        if self.model.pad_boxes:
            tlbr.append(np.array([0, 0, 0, 0]))
            num_node += 1
        else:
            if not strict:
                raise ValueError('The model is trained with no padded boxes, so it only support strict assign!')
        # extract features
        tlbr = np.array(tlbr)
        im_patches = self._crop_im_patch(tlbr=tlbr)

        #t1 = time.time()
        tlbr = torch.Tensor(tlbr).to(self.device).unsqueeze(dim=0)  # [bs, num_node, 4]
        im_shape = [self.image.size(-1), self.image.size(-2)] # [width, height]
        im_shape = torch.Tensor(im_shape).to(self.device).unsqueeze(dim=0) # [bs, 2]
        with torch.no_grad():
            app_feat = self.model.backbone(im_patches) # [num_node, feat_dim]
        app_feat = app_feat.unsqueeze(dim=0) # [bs, num_node, feat_dim]

        if match_type == 'graph_match':
            if not hasattr(self.model, 'graph_match'):
                raise ValueError('The model has no graph_match branch, please try naive_match!')

            # relative_pos: [bs, num_box, num_box, pos_dim]
            # pos_feat: [bs, num_box, num_box, pos_dim_out]
            # anchor_pos_feat: [bs, num_box, pos_dim_out]
            with torch.no_grad():
                relative_pos, pos_feat, anchor_pos_feat = self.model.graph_match.get_pos_feat(box=tlbr, im_shape=im_shape)
            # pdb.set_trace()
            neighbor_k = self.model.graph_match.get_neighbor_k(num_nodes_list=[num_node])
            pos_quantify = self.model.graph_match.pos_quantify
            if neighbor_k >= 1 and use_neighbor:
                # feat_nei: [bs, num_box, neighbor_k, feat_dim]
                # pos_feat_nei: [bs, num_box, neighbor_k, pos_dim_out]
                # tlbr_nei: [bs, num_box, neighborK, 4]
                # weight_nei: [bs, num_box, neighbor_k]
                with torch.no_grad():
                    feat_nei, pos_feat_nei, relative_pos_nei, tlbr_nei, _, weight_nei, idx_nei, _ = \
                        self.model.graph_match.pick_up_neighbors(feat=app_feat, pos_feat=pos_feat,
                                                                 relative_pos=relative_pos, box=tlbr,
                                                                 neighbor_k=neighbor_k)
            else:
                feat_nei, pos_feat_nei, relative_pos_nei, tlbr_nei, weight_nei, idx_nei = None, None, None, None, None, None

        elif match_type == 'naive_match':
            if not hasattr(self.model, 'naive_match'):
                raise ValueError('The model has no naive_match branch, please try graph_match!')
            # pos_feat: [bs, num_box, num_box, pos_dim_out]
            # anchor_pos_feat: [bs, num_box, pos_dim_out]
            with torch.no_grad():
                _, _, anchor_pos_feat = self.model.naive_match.get_pos_feat(box=tlbr, im_shape=im_shape)
            feat_nei, pos_feat_nei, relative_pos_nei, tlbr_nei, weight_nei, idx_nei = None, None, None, None, None, None
            pos_quantify = self.model.naive_match.pos_quantify
        else:
            raise ValueError('Unknown match type {}'.format(match_type))

        # pdb.set_trace()

        # set appearance features, pos embeddings, neighbors
        app_feat = app_feat.to(torch.device('cpu'))
        anchor_pos_feat = anchor_pos_feat.to(torch.device('cpu')) if anchor_pos_feat is not None else None
        relative_pos_nei = relative_pos_nei.to(torch.device('cpu')) if relative_pos_nei is not None else None
        feat_nei = feat_nei.to(torch.device('cpu')) if feat_nei is not None else None
        pos_feat_nei = pos_feat_nei.to(torch.device('cpu')) if pos_feat_nei is not None else None
        tlbr_nei = tlbr_nei.to(torch.device('cpu')) if tlbr_nei is not None else None
        weight_nei = weight_nei.to(torch.device('cpu')) if weight_nei is not None else None
        idx_nei = idx_nei.to(torch.device('cpu')) if idx_nei is not None else None # [bs, num_track, neighbor_k]

        for t_idx in range(len(tracks)):
            neighbor_tmp = {}
            neighbor_tmp['app_feat'] = feat_nei[0, t_idx] if feat_nei is not None else None# [neighbor_k, feat_dim]
            neighbor_tmp['pos_feat'] = pos_feat_nei[0, t_idx] if pos_feat_nei is not None else None
            neighbor_tmp['relative_pos'] = relative_pos_nei[0, t_idx] if relative_pos_nei is not None else None
            neighbor_tmp['tlbr'] = tlbr_nei[0, t_idx] if tlbr_nei is not None else None # [neighbor_k, 4]
            neighbor_tmp['weight'] = weight_nei[0, t_idx] if weight_nei is not None else None # [neighbor_k]
            neighbor_tmp['det_idx'] = idx_nei[0, t_idx] if idx_nei is not None else None # [neighbor_k]
            neighbor_tmp['pos_quantify'] = pos_quantify
            # change idx to identities
            if idx_nei is not None:
                if tracks_type == 'det':
                    det_ids = [tracks[idx].det_id for idx in neighbor_tmp['det_idx']]
                    neighbor_tmp['det_ids'] = det_ids
                elif tracks_type == 'track':
                    track_ids = [tracks[idx].track_id for idx in neighbor_tmp['det_idx']]
                    neighbor_tmp['track_ids'] = track_ids
                else:
                    raise ValueError('Unknown type of tracks: {}'.format(tracks_type))
            else:
                if tracks_type == 'det':
                    neighbor_tmp['det_ids'] = None
                elif tracks_type == 'track':
                    neighbor_tmp['track_ids'] = None
                else:
                    raise ValueError('Unknown type of tracks: {}'.format(tracks_type))

            tracks[t_idx].set_cur_neighbor(neighbor_tmp)

            app_feat_tmp = app_feat[0, t_idx, :] # [app_feat_dim]
            pos_feat_tmp = anchor_pos_feat[0, t_idx, :] if anchor_pos_feat is not None else None # [pos_dim_out]
            feat_tmp = {
                'app_feat': app_feat_tmp,
                'pos_feat': pos_feat_tmp,
            }
            tracks[t_idx].set_cur_feature(feature=feat_tmp)

        #t2 = time.time()
        #self.construct_time = self.construct_time + (t2 - t1)/num_node

    def _prepare_data(self, tracks, match_type, neighbor_k):
        """prepare the data to get the score

         Args:
             tracks: list of STrack
             match_type: str, which match is used
        """
        # get the data
        app_feat = []  # need to be a tensor with the size of [bs, num_track, feat_dim]
        pos_feat = []  # need to be None or a tensor with the size of [bs, num_track, pos_dim_out]
        if match_type == 'graph_match' and neighbor_k > 0:
            app_feat_nei = []  # need to be None or a tensor with the size of [bs, num_track, neighbor_k, pos_dim_out]
            pos_feat_nei = []  # need to be None or a tensor with the size of [bs, num_track, neighbor_k, pos_dim_out]
            weight_nei = []  # need to be None or a tensor with the size of [bs, num_track, neighbor_k]
        else:
            app_feat_nei, pos_feat_nei, weight_nei = None, None, None

        for t in tracks:
            app_feat_tmp = t.cur_feature['app_feat'] # 1D tensor, [feat_dim]
            app_feat.append(app_feat_tmp.unsqueeze(dim=0))

            pos_feat_tmp = t.cur_feature['pos_feat'] # None of 1D tensor, [dim_pos_out]
            if pos_feat_tmp is not None: # this means pos_feat_tmp is not None
                pos_feat.append(pos_feat_tmp.unsqueeze(dim=0))
            else: # this means pos feat is None
                pos_feat = None

            if match_type == 'graph_match' and neighbor_k > 0:
                app_feat_nei_tmp = t.cur_neighbor['app_feat'][0:neighbor_k, :] # [neighbor_k, feat_dim]
                app_feat_nei.append(app_feat_nei_tmp.unsqueeze(dim=0))

                pos_feat_nei_tmp = t.cur_neighbor['pos_feat'] # None or 2D tensor
                if pos_feat_nei_tmp is not None:
                    pos_feat_nei_tmp = pos_feat_nei_tmp[0: neighbor_k, :] # [neighbor_k, feat_dim]
                    pos_feat_nei.append(pos_feat_nei_tmp.unsqueeze(dim=0))
                else:
                    pos_feat_nei = None

                weight_nei_tmp = t.cur_neighbor['weight'][0:neighbor_k] # [neighbor_k]
                weight_nei.append(weight_nei_tmp.unsqueeze(dim=0))

        app_feat = torch.cat(app_feat, dim=0).to(self.device).unsqueeze(dim=0) # [bs, num_track, feat_dim]
        if pos_feat is not None:
            pos_feat = torch.cat(pos_feat, dim=0).to(self.device).unsqueeze(dim=0) # [bs, num_track, pos_dim_out]
        if app_feat_nei is not None:
            app_feat_nei = torch.cat(app_feat_nei, dim=0).to(self.device).unsqueeze(dim=0) # [bs, num_track, neighbor_k, feat_dim]
        if pos_feat_nei is not None:
            pos_feat_nei = torch.cat(pos_feat_nei, dim=0).to(self.device).unsqueeze(dim=0) # [bs, num_track, neighbor_k, pos_dim_out]
        if weight_nei is not None:
            weight_nei = torch.cat(weight_nei, dim=0).to(self.device).unsqueeze(dim=0) # [bs, num_track, neighbor_k]

        return app_feat, pos_feat, app_feat_nei, pos_feat_nei, weight_nei

    def get_dist_matrix(self, tracks, detections, match_type='graph_match', use_neighbor=True):

        """Get the dist matrix between the given boxes in two frames

         Args:
             tracks: list of STrack
             detections: list of STrack
             match_type: str, which match is used
        """
        num_track = len(tracks)
        num_det = len(detections)
        if num_track == 0 or num_det == 0:
            return np.zeros((num_track, num_det), dtype=np.float)

        # get the neighbor_k
        if match_type == 'graph_match' and use_neighbor:
            neighbor_k = 1e3
            for t in itertools.chain(tracks, detections):
                app_feat_nei_tmp = t.cur_neighbor['app_feat']  # None or a 2D tensor with the size [neighbor_k, feat_dim]
                if app_feat_nei_tmp is None:
                    neighbor_k = 0
                    break
                else:
                    neighbor_k = min(neighbor_k, app_feat_nei_tmp.size(0))
        else:
            neighbor_k = -1

        # prepare data to get the score
        # app_feat_t: # [bs, num_track, feat_dim]
        # pos_feat_t: # [bs, num_track, pos_dim_out]
        # app_feat_nei_t: [bs, num_track, neighbor_k, feat_dim]
        # pos_feat_nei_t: [bs, num_track, neighbor_k, pos_dim_out]
        # weight_t: [bs, num_track, neighbor_k]
        t1_c = time.time()
        app_feat_t, pos_feat_t, app_feat_nei_t, pos_feat_nei_t, weight_nei_t = self._prepare_data(tracks=tracks, match_type=match_type, neighbor_k=neighbor_k)
        app_feat_d, pos_feat_d, app_feat_nei_d, pos_feat_nei_d, weight_nei_d = self._prepare_data(tracks=detections, match_type=match_type, neighbor_k=neighbor_k)
        t2_c = time.time()
        self.construct_time = self.construct_time + (t2_c - t1_c)/(num_track+num_det)


        t1_s = time.time()
        if match_type == 'graph_match':
            if not hasattr(self.model, 'graph_match'):
                raise ValueError('The model has no graph_match branch, please try naive_match!')
            with torch.no_grad():
                # ==== get the score in a loop, so that it will not out of memory ====
                total_score = []
                batch_track = int(2000 / app_feat_d.size(1))
                # batch_track = 40
                num_track = app_feat_t.size(1)
                loops = num_track // batch_track
                if batch_track * loops < num_track:
                    loops += 1

                for l in range(loops):
                    idx1 = l * batch_track
                    idx2 = min((l+1)*batch_track, num_track)

                    app_feat_t_tmp = app_feat_t[:, idx1:idx2, :]
                    pos_feat_t_tmp = pos_feat_t[:, idx1:idx2, :] if pos_feat_t is not None else None
                    app_feat_nei_t_tmp = app_feat_nei_t[:, idx1:idx2, :, :] if app_feat_nei_t is not None else None
                    pos_feat_nei_t_tmp = pos_feat_nei_t[:, idx1:idx2, :, :] if pos_feat_nei_t is not None else None
                    weight_nei_t_tmp = weight_nei_t[:, idx1:idx2, :] if weight_nei_t is not None else None

                    score_tmp = self.model.graph_match.get_score(feat1=app_feat_t_tmp,
                                                                 pos_feat1=pos_feat_t_tmp,
                                                                 feat_nei1=app_feat_nei_t_tmp,
                                                                 pos_feat_nei1=pos_feat_nei_t_tmp,
                                                                 weight_nei1=weight_nei_t_tmp,
                                                                 feat2=app_feat_d, pos_feat2=pos_feat_d, feat_nei2=app_feat_nei_d,
                                                                 pos_feat_nei2=pos_feat_nei_d, weight_nei2=weight_nei_d) # tuple
                    total_score.append(score_tmp[0])
                score = torch.cat(total_score, dim=1) # [bs, num_track, num_det]

                # ==== get the score in one forward ====
                # score = self.model.graph_match.get_score(feat1=app_feat_t, pos_feat1=pos_feat_t, feat_nei1=app_feat_nei_t,
                #                                          pos_feat_nei1=pos_feat_nei_t, weight_nei1=weight_nei_t,
                #                                          feat2=app_feat_d, pos_feat2=pos_feat_d, feat_nei2=app_feat_nei_d,
                #                                          pos_feat_nei2=pos_feat_nei_d, weight_nei2=weight_nei_d) # tuple
                # # score, score_anchor, score_nei = score
                # score = score[0]# [bs, num_track, num_det]

        elif match_type == 'naive_match':
            if not hasattr(self.model, 'naive_match'):
                raise ValueError('The model has no naive_match branch, please try graph_match!')
            with torch.no_grad():
                score = self.model.naive_match.get_score(feat1=app_feat_t, pos_feat1=pos_feat_t,
                                                         feat2=app_feat_d, pos_feat2=pos_feat_d) # [bs, num_track, num_det]
        t2_s = time.time()
        self.score_time = self.score_time + (t2_s-t1_s) / (num_track * num_det)
        # print(t2_s-t1_s, num_track, num_det)
        score = score.squeeze(0) # [num_track, num_det]
        dist = 1 - score
        dist = dist.to(torch.device('cpu')).numpy()

        return dist

    def association(self, tracks, detections, match_type='graph_match', threshold=np.inf, gate=True,
                    kalman_filter=None, use_neighbor=True):

        """Get the dist matrix between the given boxes in two frames

         Args:
             tracks: list of STrack
             detections: list of STrack
             match_type: str, which match is used
             threshold: float or None, if float, the cost that larger than it will not be associated.
                Otherwise, we will directly associated tracks with detection.
            kalman_filter: The filter used to gate the cost. If None, we will not gate the cost matrix
                based on euclidean.
            gate: bool, whether to gate the cost matrix
            use_neighbor: bool, whether to use neighbor information
        """
        num_track = len(tracks)
        num_det = len(detections)
        if num_track == 0 or num_det == 0:
            matches = []
            un_matched_t = list(range(num_track))
            un_matched_d = list(range(num_det))
            return matches, un_matched_t, un_matched_d

        if self.debug:
            debug_frame = 10

            image = self.image2plot(channel='BGR')

            # plot
            # detections
            plt.clf()
            tlwh = [t.tlwh() for t in tracks]
            ids = list(range(len(tracks)))
            track_im = vis.plot_tracking(image=image.copy(), tlwhs=tlwh, obj_ids=ids, frame_id=self.frame_id)
            track_im = cv2.cvtColor(track_im, cv2.COLOR_BGR2RGB)
            if self.frame_id >= debug_frame:
                plt.subplot(2,3,1)
                plt.title('track')
                plt.imshow(track_im)

            tlwh = [d.tlwh() for d in detections]
            ids = [i for i, d in enumerate(detections)]
            det_im = vis.plot_tracking(image=image.copy(), tlwhs=tlwh, obj_ids=ids, frame_id=self.frame_id)
            det_im = cv2.cvtColor(det_im, cv2.COLOR_BGR2RGB)
            if self.frame_id >= debug_frame:
                # plt.imsave('./shot_images/frame_{}.png'.format(str(self.frame_id)), det_im)
                plt.subplot(2,3,2)
                plt.title('detection')
                plt.imshow(det_im)

            if self.frame_id >= debug_frame:
                if len(self.det_frames) >= 1:
                    plt.subplot(2, 3, 3)
                    plt.title('detection')
                    # plt.imsave('./shot_images/frame_{}.png'.format(str(self.frame_id-1)), self.det_frames[-1])
                    plt.imshow(self.det_frames[-1])
                if len(self.det_frames) >= 2:
                    plt.subplot(2, 3, 4)
                    plt.title('detection')
                    plt.imshow(self.det_frames[-2])
                if len(self.det_frames) >= 3:
                    plt.subplot(2, 3, 5)
                    plt.title('detection')
                    plt.imshow(self.det_frames[-3])
                if len(self.det_frames) >= 4:
                    plt.subplot(2, 3, 6)
                    plt.title('detection')
                    plt.imshow(self.det_frames[-4])
                plt.subplots_adjust(left=0.03, top=0.97, right=0.97, bottom=0.03, wspace=0.08, hspace=0.03)
                plt.pause(0.1)
                a = 1

            if self.pre_frame_id != self.frame_id:
                self.det_frames.append(det_im)
            self.pre_frame_id = self.frame_id

        # if strict: cost has size [num_track, num_det]
        # else: cost has size [N, N], where N = max(num_track+1, num_det+1)
        cost = self.get_dist_matrix(tracks=tracks, detections=detections,
                                    match_type=match_type, use_neighbor=use_neighbor)
        if gate:
            # TODO: if strict is False, then the gating my not be accurate
            cost = gate_cost_matrix(kf=kalman_filter, cost_matrix=cost, tracks=tracks, detections=detections)
        cost[cost > threshold] = threshold + 1e-5

        assigned_idx, _, _ = self.model._hungarian_assign(cost_matrix=cost, strict=True, tool='lapjv')
        # assigned_idx: [num_assigine, 2], the first col is the index of tracks,
        # the second row is the index of detections
        matches, unmatched_t, unmatched_d = indices_to_matches(cost_matrix=cost, indices=assigned_idx, thresh=threshold)

        return matches, unmatched_t, unmatched_d

    def find_lost_tracks(self, frame_id, lost_tracks, support_tracks, match_type='graph_match',
                         strict=True, use_neighbor=True, threshold=0.):
        """This function get the neighbors for detections. Once a detection is matched with a
        track, the information of neighbors will be delivered to tracks.

        Args:
            lost_tracks: tracks: list of lost STrack
            support_tracks: list of Stracks, which are used to find the lost tracks
            match_type: str, which match is used
            strict: bool, if true, we will perform association between true tracks and true detections.
                If False, we will do association between all (including padded)tracks and all (including
                padded) detections.
            use_neighbor: bool whether to use neighbors when perform graph match. Only effective in 'graph_match'
            threshold: float, the min distance used to find the lost tracks
        """
        num_lost = len(lost_tracks)
        num_sup = len(support_tracks)
        if num_lost == 0 or num_sup == 0:
            return

        if match_type != 'graph_match':
            return

        neighbor_k = self.model.graph_match.get_neighbor_k([num_sup + 1])  # plus 1 if for the candidate
        if neighbor_k < 1 or not use_neighbor:
            return

        # note that the features and neighbors of lost tracks are not updated
        # while the features and neighbors of support tracks should be updated

        # prepare im shape
        im_shape = [self.image.size(-1), self.image.size(-2)] # [width, height]
        im_shape = torch.Tensor(im_shape).to(self.device) # [2]
        im_shape = im_shape.view(1, 2) # [bs, 2]

        # get the box and app feature of support tracks
        tlbr_s = []
        app_feat_s = []
        for st in support_tracks:
            tlbr_s.append(st.tlbr()) # each element is an array
            app_feat_s.append(st.cur_feature['app_feat'].unsqueeze(0)) # 1D tensor, [feat_dim]
        # pad box
        if self.model.pad_boxes:
            raise NotImplementedError
        else:
            if not strict:
                raise ValueError('The model is trained with no padded boxes, so it only support strict assign!')
        tlbr_s = torch.Tensor(tlbr_s).to(self.device) # [num_sup, 4]
        app_feat_s = torch.cat(app_feat_s, dim=0).to(self.device) # [num_sup, app_feat_dim]

        # forward to get the app features and position features for lost tracks
        valid_lost_tracks = []
        for lt in lost_tracks:
            if not lt.has_neighbor():
                continue
            valid_lost_tracks.append(lt)

        # batch_num = 4
        # batch_num = int((20000 / (len(support_tracks) + 1)) / 65)
        batch_num = int((15000 / (len(support_tracks) + 1)) / 65)
        num_valid_lost = len(valid_lost_tracks)
        loops = num_valid_lost // batch_num
        if loops * batch_num < num_valid_lost:
            loops += 1

        for l in range(loops):
            lost_tracks_filter = valid_lost_tracks[l*batch_num:min((l+1)*batch_num, num_valid_lost)]

            num_candidate = []
            tlbr_candidate = []
            for lt in lost_tracks_filter:
                # prepare the information of lost track in current frame candidate
                candidate = lt.sample_candidate(format='tlbr') # [num_candicate, 4]
                tlbr_candidate.append(candidate)

                num_can = candidate.shape[0]
                num_candidate.append(num_can)

            # get app feature
            tlbr_candidate = np.concatenate(tlbr_candidate, axis=0)
            tlbr_candidate = torch.Tensor(tlbr_candidate).to(self.device)
            patches = self._crop_im_patch(tlbr=tlbr_candidate)
            with torch.no_grad():
                app_feat_can = self.model.backbone(patches)  # [num_candidate, feat_dim]

            # get pos feature
            tlbr_s_tmp = tlbr_s.view(1, num_sup, 4).repeat(tlbr_candidate.size(0), 1, 1) # [num_candidate, num_sup, 4]
            tlbr_can = tlbr_candidate.view(tlbr_candidate.size(0), 1, 4) # [num_candidate, 1, 4]
            tlbr_can = torch.cat((tlbr_s_tmp, tlbr_can), dim=1) # [num_candidate, num_sup+1, 4]
            # pos_feat_cur: [num_candidate, num_sup+1, num_sup+1, pos_feat_dim]
            # anchor_pos_feat_cur: [num_candidate, num_sup+1, pos_feat_dim]
            with torch.no_grad():
                #print('num_candicate: {}, num_support {}'.format(tlbr_can.size(0), tlbr_can.size(1)))
                relative_pos_can, pos_feat_can, anchor_pos_feat_can = self.model.graph_match.get_pos_feat(box=tlbr_can, im_shape=im_shape)

            # handle the lost tracks
            idx1 = 0
            idx2 = 0
            for i in range(len(lost_tracks_filter)):
                lt = lost_tracks_filter[i]

                num_can = num_candidate[i]
                idx2 = idx2+num_can

                # prepare neighbor information of lost track in previous frame
                app_feat_pre = lt.cur_feature['app_feat'] # 1D tensor, [app_feat_dim]
                anchor_pos_feat_pre = lt.cur_feature['pos_feat'] # 1D tensor, [pos_feat_dim]

                app_feat_nei_pre = lt.cur_neighbor['app_feat'] # [neighbor_k, app_feat_dim] or None
                pos_feat_nei_pre = lt.cur_neighbor['pos_feat'] # [neighbor_k, pos_feat_dim] or None
                weight_nei_pre = lt.cur_neighbor['weight'] # [neighbor_k] or None

                neighbor_k_tmp = min(neighbor_k, app_feat_nei_pre.size(0))
                if neighbor_k_tmp < 1:
                    continue

                # prepare the information of lost track in current frame candidate
                # get app feature
                app_feat_cur = app_feat_can[idx1:idx2]  # [num_candidate, feat_dim]
                app_feat_s_tmp = app_feat_s.view(1, num_sup, -1).repeat(num_can, 1, 1) # [num_candidate, num_sup, app_feat_dim]

                app_feat_cur = app_feat_cur.view(num_can, 1, -1) # [num_candidate, 1, app_feat_dim]
                app_feat_cur = torch.cat((app_feat_s_tmp, app_feat_cur), dim=1) # [num_candidate, num_sup+1, app_feat_dim]

                # get pos feature
                tlbr_cur = tlbr_can[idx1:idx2] # [num_candidate, num_sup+1, 4]
                relative_pos_cur = relative_pos_can[idx1:idx2] # [num_candidate, num_sup+1, num_sup+1, 8]
                pos_feat_cur = pos_feat_can[idx1:idx2] # [num_candidate, num_sup+1, num_sup+1, pos_feat_dim]
                anchor_pos_feat_cur = anchor_pos_feat_can[idx1:idx2] # [num_candidate, num_sup+1, pos_feat_dim]

                # app_feat_nei: [num_candidate, num_sup+1, neighbor_k, app_feat_dim]
                # pos_feat_nei: [num_candidate, num_sup+1, neighbor_k, app_feat_dim]
                # tlbr_nei: [num_candidate, num_sup+1, neighbor_k, 4]
                # weight_nei: [num_candidate, num_sup+1, neighbor_k], the weight to get the neighbors
                # feat_nei, pos_feat_nei, relative_pos_nei, box_nei, ids_nei, nei_v, nei_idx, weight_logits
                app_feat_nei_cur, pos_feat_nei_cur, _, tlbr_nei_cur, _,  weight_nei_cur, _, _ = \
                    self.model.graph_match.pick_up_neighbors(feat=app_feat_cur, pos_feat=pos_feat_cur,
                                                             relative_pos=relative_pos_cur,
                                                             box=tlbr_cur, neighbor_k=neighbor_k_tmp)

                # prepare the data to get the score
                app_feat_cur = app_feat_cur[:, -1:, :] # [num_candidate, 1, app_feat_dim]
                anchor_pos_feat_cur = anchor_pos_feat_cur[:, -1:, :] if anchor_pos_feat_cur is not None else None # [num_candidate, 1, pos_feat_dim]
                app_feat_nei_cur = app_feat_nei_cur[:, -1:, :, :] # [num_candidate, 1, neighbor_k, app_feat_dim]
                pos_feat_nei_cur = pos_feat_nei_cur[:, -1:, :, :] if pos_feat_nei_cur is not None else None # [num_candidate, 1, neighbor_k, pos_feat_dim]
                tlbr_nei_cur = tlbr_nei_cur[:, -1, :, :] if tlbr_nei_cur is not None else None # [num_candidate, 1, neighbor_k, 4]
                weight_nei_cur = weight_nei_cur[:, -1:, :] if weight_nei_cur is not None else None # [num_candidate, 1, neighbor_k]

                app_feat_pre = app_feat_pre.view(1, 1, -1).repeat(num_can, 1, 1).to(self.device)  # [num_candidate, 1, app_feat_dim]
                if anchor_pos_feat_pre is not None:
                    anchor_pos_feat_pre = anchor_pos_feat_pre.to(self.device)
                    anchor_pos_feat_pre = anchor_pos_feat_pre.view(1, 1, anchor_pos_feat_pre.size(-1)).repeat(num_can, 1, 1) # [num_candidate, 1, pos_feat_dim]
                if app_feat_nei_pre is not None:
                    app_feat_nei_pre = app_feat_nei_pre[0:neighbor_k_tmp].to(self.device) # [neighbor_k, app_feat_dim]
                    app_feat_nei_pre = app_feat_nei_pre.view(1, 1, neighbor_k_tmp, app_feat_nei_pre.size(-1)).repeat(num_can, 1, 1, 1)  # [num_candidate, 1, neighbor_k, app_feat_dim]
                if pos_feat_nei_pre is not None:
                    pos_feat_nei_pre = pos_feat_nei_pre[0:neighbor_k_tmp].to(self.device) # [neighbor_k, pos_feat_dim]
                    pos_feat_nei_pre = pos_feat_nei_pre.view(1, 1, neighbor_k_tmp, pos_feat_nei_pre.size(-1)).repeat(num_can, 1, 1, 1) # [num_candidate, 1, neighbor_k, pos_feat_dim]
                if weight_nei_pre is not None:
                    weight_nei_pre = weight_nei_pre[0:neighbor_k_tmp].to(self.device) # [neighbor_k]
                    weight_nei_pre = weight_nei_pre.view(1, 1, weight_nei_pre.size(-1)).repeat(num_can, 1, 1) # [num_candidate, 1, neighbor_k]
                with torch.no_grad():
                    score = self.model.graph_match.get_score(feat1=app_feat_pre, pos_feat1=anchor_pos_feat_pre, feat_nei1=app_feat_nei_pre,
                                                       pos_feat_nei1=pos_feat_nei_pre, weight_nei1=weight_nei_pre,
                                                       feat2=app_feat_cur, pos_feat2=anchor_pos_feat_cur, feat_nei2=app_feat_nei_cur,
                                                       pos_feat_nei2=pos_feat_nei_cur, weight_nei2=weight_nei_cur) # tuple

                score = score[0]  # [num_candidate, 1,1]
                dist = 1 - score.squeeze(-1).squeeze(-1) # num_candidate
                min_dist, min_idx = torch.topk(dist, k=1, largest=False)
                if min_dist.item() < threshold * 0.7:

                    tracked = min_dist.item() < threshold

                    # prepare anchor feature
                    feat = {
                        'app_feat': app_feat_cur[min_idx.item()].squeeze(dim=0),
                        'pos_feat': anchor_pos_feat_cur[min_idx.item()].squeeze(dim=0),
                    }

                    # prepare box
                    box = tlbr_cur[:, -1, :].view(num_can, 4)[min_idx.item()].to(torch.device('cpu')).numpy() # tlbr
                    box[2:4] = box[2:4] - box[0:2]

                    # prepare neighbor
                    neighbor = {
                        'app_feat': app_feat_nei_cur[min_idx.item()].squeeze(dim=0),
                        'pos_feat': pos_feat_nei_cur[min_idx.item()].squeeze(dim=0) if pos_feat_nei_cur is not None else None,
                        'weight': weight_nei_cur[min_idx.item()].squeeze(dim=0) if weight_nei_cur is not None else None,
                        'tlbr': tlbr_nei_cur[min_idx.item()].squeeze(dim=0) if tlbr_nei_cur is not None else None
                    }

                    lt.re_find(tlwh=box, feature=feat, neighbor=neighbor, frame_id=frame_id, tracked=tracked)

                idx1 = idx2


if __name__ == '__main__':

    a = np.array([1,2,3])
    b = np.array([4,5,6])
    d = []
    d.append(a)
    d.append(b)
    c = torch.Tensor(d)

    print(c)





















