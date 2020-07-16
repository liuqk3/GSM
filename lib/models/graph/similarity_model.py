import torch
import torch.nn as nn
import numpy as np
import lapjv
from lib.models.backbone.resnet import ResNetBackbone
from lib.models.reid.reid_model import ReIDModel
from scipy.optimize import linear_sum_assignment
from .naive_match import NaiveMatch
from .graph_match_v5 import GraphMatch as GraphMatch_v5

class GraphSimilarityl(nn.Module):
    def __init__(self, backbone_name, backbone_args, graphmatch_args,
                 naivematch_args, train_part='all', pad_boxes=False,
                 match_name='GraphMatch'):
        """This class get the similarity score between detections and tracks

        Args:
            backbone_name: str, the name of backbone name
            backbone_args: dict, the args to initialize the backbone
            graphmatch_args: dict, the args to the GraphMatch module
            naivematch_args: dict, the args to the NaiveMatch module
            train_part: str, which part to train
            pad_boxes: bool, whether the inputed data is padded a track box and a det box
            match_name: str, 'GraphMatch', 'NaiveMatch', or 'both'
        """
        super(GraphSimilarityl, self).__init__()
        # pdb.set_trace()
        self.name = 'GraphSimilarity'

        self.backbone_name = backbone_name
        self.train_part = 'backbone, graph_match, naive_match' if train_part == 'all' else train_part
        self.pad_boxes = pad_boxes

        self.match_name = match_name

        if self.backbone_name == 'ReIDModel':
            self.backbone = ReIDModel(**backbone_args)
        elif self.backbone_name == 'ResNetBackbone':
            self.backbone = ResNetBackbone(**backbone_args)

        match_names = self.match_name.split(',')
        if 'GraphMatch_v5' in match_names:
            self.graph_match = GraphMatch_v5(**graphmatch_args)

        if 'NaiveMatch' in match_names:
            self.naive_match = NaiveMatch(**naivematch_args)

        self.train()

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            if 'backbone' in self.train_part:
                print('set backbone trainable...')
                self.backbone.train()
            else:
                for key, value in dict(self.backbone.named_parameters()).items():
                    value.requires_grad = False
                self.backbone.eval()

            if 'GraphMatch' in self.match_name:
                if 'graph_match' in self.train_part:
                    print('set GraphMatch trainable...')
                    self.graph_match.train()
                else:
                    for key, value in dict(self.graph_match.named_parameters()).items():
                        value.requires_grad = False
                    self.graph_match.eval()

            if 'NaiveMatch' in self.match_name:
                if 'naive_match' in self.train_part:
                    self.naive_match.train()
                    print('set NaiveMatch trainable...')
                else:
                    for key, value in dict(self.naive_match.named_parameters()).items():
                        value.requires_grad = False
                    self.naive_match.eval()

    def _hungarian_assign(self, cost_matrix, ids1=None, ids2=None, strict=False, tool='lapjv'):
        """This function perform data association based on hungarian algorithm

        Args:
            cost_matrix: 2D tensor, [n, m]
            ids1: 1D tensor, [n], the gt ids for the first frame
            ids2: 1D tensor, [m], the gt ids for the second frame. If both ids1 and ids2 are None,
                this function may be called while tracking online
            strict: bool, if true, we only try to assign true detections with true tracks.
                Otherwise, we try to assing all (including padded) detections to all
                (included padded) tracks.
        """

        if isinstance(cost_matrix, torch.Tensor):
            cost_matrix = cost_matrix.numpy()
        if isinstance(ids1, torch.Tensor):
            ids1 = ids1.numpy()
        if isinstance(ids2, torch.Tensor):
            ids2 = ids2.numpy()

        if not strict and not self.pad_boxes:
            raise ValueError('There are no padded boxes in the input data, only support strict assign!')

        if ids1 is not None and ids2 is not None:
            if strict:
                # filter out empty and padded nodes
                if ids1 is not None:
                    index1 = ids1 > 0
                    ids1 = ids1[index1]
                    cost_matrix = cost_matrix[index1, :]
                if ids2 is not None:
                    index2 = ids2 > 0
                    ids2 = ids2[index2]
                    cost_matrix = cost_matrix[:, index2]
            else:
                # filter out empty nodes
                # filter out empty and padded nodes
                if ids1 is not None:
                    index1 = ids1 >= 0
                    ids1 = ids1[index1]
                    cost_matrix = cost_matrix[index1, :]
                if ids2 is not None:
                    index2 = ids2 >= 0
                    ids2 = ids2[index2]
                    cost_matrix = cost_matrix[:, index2]

                # the last row is the padded track, and the last column is padded detection.
                # in order to make sure that each node will be assigned anthoer node, we pad
                # the cost matrix into a square matrix
                num_track = cost_matrix.shape[0]
                num_det = cost_matrix.shape[1]
                if num_track > num_det:
                    # repeat the last column
                    cost_col = cost_matrix[:, -1]
                    cost_col = cost_col[:, np.newaxis] # [num_track, 1]
                    cost_col = np.tile(cost_col, (1, num_track-num_det))
                    cost_matrix = np.concatenate((cost_matrix, cost_col), axis=1)
                    if ids2 is not None:
                        ids2 = np.concatenate((ids2, np.tile(ids2[-1], (num_track-num_det))), axis=0)
                elif num_det > num_track:
                    # repeat the last tow
                    cost_row = cost_matrix[-1, :]
                    cost_row = cost_row[np.newaxis, :]
                    cost_row = np.tile(cost_row, (num_det-num_track, 1))
                    cost_matrix = np.concatenate((cost_matrix, cost_row), axis=0)
                    if ids1 is not None:
                        ids1 = np.concatenate((ids1, np.tile(ids1[-1], (num_det-num_track))), axis=0)

        if tool == 'scipy':
            indices = linear_sum_assignment(cost_matrix)
            if isinstance(indices, tuple):
                indices = np.array(indices).transpose() # [num_assign, 2], the first row is the index of tracks, and the second is the index of detections
        elif tool == 'lapjv':
            # pad the cost matrix to a square matrix for tool lapjv
            num_track, num_det = cost_matrix.shape
            if num_track > num_det:
                pad = np.zeros((num_track, num_track - num_det))
                pad.fill(1e5)
                cost_matrix = np.concatenate((cost_matrix, pad), axis=1)

            elif num_det > num_track:
                pad = np.zeros((num_det - num_track, num_det))
                pad.fill(1e5)
                cost_matrix = np.concatenate((cost_matrix, pad), axis=0)
            ass = lapjv.lapjv(cost_matrix)
            r_ass, c_ass = ass[0], ass[1]
            if num_track <= num_det:
                r_ass = r_ass[0:num_track]
                indices = np.arange(0, num_track)
                indices = np.array([indices, r_ass]).transpose() # [num_assign, 2]
            else:
                c_ass = c_ass[0:num_det]
                indices = np.arange(0, num_det)
                indices = np.array([c_ass, indices]).transpose() # [num_assign, 2]

        if ids1 is not None and ids2 is not None:
            ids1_tmp = ids1[indices[:, 0]]
            ids2_tmp = ids2[indices[:, 1]]

            right_num = ids1_tmp == ids2_tmp
            right_num = right_num.sum()
            assign_num = indices.shape[0]

            if not strict: # we need to check the disappeared track and appeared detection
                ids1_true = ids1[ids1 > 0]
                ids2_true = ids2[ids2 > 0]

                mask = ids1_true[:, np.newaxis] == ids2_true[np.newaxis, :] # [num_true_track, num_true_det]

                # check if there are some tracks disappear
                disappear_idx = np.sum(mask, axis=1) == 0 # [num_true_track]
                ids1_disappear = ids1_true[disappear_idx] # [num_disappear]
                ids1_assign_disappear = ids1_tmp[ids2_tmp==0] # [num_assign_disappear]

                disappear_mask = ids1_assign_disappear[:, np.newaxis] == ids1_disappear[np.newaxis, :]
                disappear_idx = np.sum(disappear_mask, axis=1) > 0 # [num_assign_disappear]
                num_true_disappear = disappear_idx.sum()

                # check if there are som detect appear
                appear_idx = np.sum(mask, axis=0) == 0# [num_ture_det]
                ids2_appear = ids2_true[appear_idx]
                ids2_assign_appear = ids2_tmp[ids1_tmp == 0]
                appear_mask = ids2_assign_appear[:, np.newaxis] == ids2_appear[np.newaxis, :]
                appear_idx = np.sum(appear_mask, axis=1) > 0 # [num_assign_disappear]
                num_true_appear = appear_idx.sum()

                right_num = right_num + num_true_disappear + num_true_appear
            assert right_num <= assign_num
        else:
            assign_num, right_num = 0, 0

        return indices, assign_num, right_num

    def _get_acc(self, score, obj_id1, obj_id2):
        """get the accuracy using Hungarian algorithm

        Args:
            score: 3D tensor, [bs, num_node1, num_node2]
            obj_id1: 2D tensor, [bs, num_node1], the id of track node
            obj_id2: 2D tensor, [bs, num_node2], the if of det node
        """
        if obj_id1 is None and obj_id2 is None:
            acc = torch.Tensor([0]).to(score.device)
        else:
            bs = score.size(0)
            cost = 1 - score

            assign_num = []
            right_num = []
            cost_cpu = cost.detach().cpu()
            track_id_cpu = obj_id1.cpu()
            det_id_cpu = obj_id2.cpu()
            for b in range(bs):
                _, assign_num_tmp, right_num_tmp = self._hungarian_assign(cost_matrix=cost_cpu[b], ids1=track_id_cpu[b],
                                                                          ids2=det_id_cpu[b], strict=True)
                assign_num.append(assign_num_tmp)
                right_num.append(right_num_tmp)

            assign_num = sum(assign_num)
            right_num = sum(right_num)

            acc = right_num / assign_num
            # pdb.set_trace()
            acc = torch.Tensor([acc]).to(score.device)

        return acc

    def forward(self, im_patch1, im_patch2, obj_id1=None, obj_id2=None, box1=None, box2=None, im_shape=None):
        """get the similarity between the tracks and detections

        Args:

            im_patch1: 4D tensor, [bs, num_node1, 3, height, width]
            im_patch2: 4D tensor, [bs, num_node2, 3, height, width]
            obj_id1: 2D tensor, [bs, num_node1], the id of track node
            obj_id2: 2D tensor, [bs, num_node2], the if of det node
            box1: 3D tensor, [bs, num_node1, 4], each box is presented as [x1, y1, x2, y2]
            box1: 3D tensor, [bs, num_node2, 4], each box is presented as [x1, y1, x2, y2]
            im_shape: 2D tensor, [bs, 2], [width, height]
        """

        bs, num_node1, channel, height, width = im_patch1.size()
        num_node2 = im_patch2.size(1)

        all_patch = torch.cat((im_patch1, im_patch2), dim=1) # [bs, num_node1+num_node2, channel, height, width]
        feat = self.backbone(all_patch.view(bs * (num_node1+num_node2), channel, height, width))

        feat = feat.view(bs, num_node1+num_node2, -1)
        feat1, feat2 = torch.split(feat, split_size_or_sections=[num_node1, num_node2], dim=1)

        loss = {}
        acc = {}

        # if we have the graph match
        if 'GraphMatch' in self.match_name:
            if self.graph_match.name in ['GraphMatch_v5']:
                score_g, loss_g = self.graph_match(feat1=feat1, feat2=feat2, box1=box1,
                                                   box2=box2, obj_id1=obj_id1, obj_id2=obj_id2, im_shape=im_shape)
                for k in loss_g.keys():
                    loss[k + '_graph'] = loss_g[k]

                acc_g = self._get_acc(score=score_g['score_g'], obj_id1=obj_id1, obj_id2=obj_id2)
                acc_ga = self._get_acc(score=score_g['score_a'], obj_id1=obj_id1, obj_id2=obj_id2)
                acc['acc_g_graph'] = acc_g
                acc['acc_a_graph'] = acc_ga

        if 'NaiveMatch' in self.match_name:
            score_n, loss_n = self.naive_match(feat1=feat1, feat2=feat2, box1=box1, box2=box2,
                                               obj_id1=obj_id1, obj_id2=obj_id2, im_shape=im_shape)
            acc_n = self._get_acc(score=score_n['score'], obj_id1=obj_id1, obj_id2=obj_id2)
            for k in loss_n.keys():
                loss[k+'_naive'] = loss_n[k]
            acc['acc_naive'] = acc_n

        return loss, acc
