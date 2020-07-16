import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict
import lapjv

def binary_cross_entropy_loss(score, gt_score, ratio=-1):
    """Get the binary cross entropy loss
    Args:
        score: multi-dim tensor, predicted score
        gt_score: multi-dim tensor, ground-truth score, 1 is positive, 0 is negative
        ratio: the ratio number of negative and positive samples, if -1, we use all negative samples
    """
    # positive samples
    margin = 1e-10
    mask = gt_score == 1
    score_pos = score[mask]  # 1D tensor
    score_pos = score_pos[score_pos > margin]
    #score_pos[score_pos < margin] = score_pos[score_pos < margin] * 0 + margin
    loss_pos = 0 - torch.log(score_pos)
    num_pos = loss_pos.size(0)

    # negative samples
    mask = gt_score == 0
    score_neg = score[mask]  # 1D tensor
    score_neg = 1 - score_neg
    score_neg = score_neg[score_neg > margin]
    #score_neg[score_neg < margin] = score_neg[score_neg < margin] * 0 + margin
    loss_neg = 0 - torch.log(score_neg)
    num_neg = loss_neg.size(0)
    if ratio > 0:
        loss_neg = loss_neg.sort(descending=True)[0]
        num_neg = min(loss_neg.size(0), int(ratio*num_pos))
        loss_neg = loss_neg[0:num_neg]

    if num_neg > 0 and num_pos > 0:
        weight_pos = num_neg / (num_pos + num_neg)
        weight_neg = num_pos / (num_pos + num_neg)
        loss = weight_pos * loss_pos.mean() + weight_neg * loss_neg.mean()
    elif num_neg == 0 and num_pos > 0:
        loss = loss_pos.mean()
    elif num_neg > 0 and num_pos == 0:
        loss = loss_neg.mean()
    else:
        loss = score.mean() * 0 # torch.Tensor([0]).to(score.device)

    return loss


def association_neighbor(score, threshold=None, tool='hungarian'):
    """This function association the neighbors of two anchors

    Args:
        score: [bs, num_node1, num_node2, neighbor_k, neighbor_k]
        threshold: float or None, the score that lower than this threshold will
            not be associated.
    """
    if threshold is not None:
        raise NotImplementedError


    if tool in ['scipy', 'lapjv']:
        bs, num_node1, num_node2, neighbor_k, _ = score.size()
        dist = score.view(-1, neighbor_k, neighbor_k)
        dist = 1 - dist
        dist = dist.detach().to(torch.device('cpu')).numpy()

        if tool == 'scipy':
            index1, index2 = [], []
            for i in range(dist.shape[0]):
                r_idx, c_idx = linear_sum_assignment(dist[i])
                index1.append(r_idx)
                index2.append(c_idx)

            index1 = torch.Tensor(index1).view(bs, num_node1, num_node2, neighbor_k, 1).to(score.device).long()
            index2 = torch.Tensor(index2).view(bs, num_node1, num_node2, neighbor_k, 1).to(score.device).long()

            index1 = index1.repeat(1, 1, 1, 1, neighbor_k)
            score = torch.gather(score, index=index1, dim=-2)  # [bs, num_node1, num_node2, neighbor_k, neighbor_k]
            score = torch.gather(score, index=index2, dim=-1)  # [bs, num_node1, num_node2, neighbor_k, 1]

        elif tool == 'lapjv':
            ass1 = []
            for i in range(dist.shape[0]):
                # r_ass = hungarian.lap(dist[i])[0]
                r_ass = lapjv.lapjv(dist[i])[0]
                ass1.append(r_ass)

            ass1 = torch.Tensor(ass1).view(bs, num_node1, num_node2, neighbor_k, 1).to(score.device).long() #  [bs, num_node1, num_node2, neighbor_k, 1]
            score = torch.gather(score, index=ass1, dim=-1)  # [bs, num_node1, num_node2, neighbor_k, 1]

        score = score.view(bs, num_node1, num_node2, neighbor_k)  # [bs, num_node1, num_node2, neighbor_k]
    elif tool == 'min':
        score, _ = score.min(dim=-2)

    return score


def arrange_neighbor(neighbor1, neighbor2, assignment):
    """This function re-arrange the neighbor, so the neighbors will be matched

    Args:
        neighbor1: 3D tensor, [bs, num_node1, neighbor_k, dim]
        neighbor2: 3D tensor, [bs, num_node2, neighbor_k, dim]
        assignment: [bs, num_node1, num_node2, neighbor_k, 2]
    """
    index1, index2 = torch.chunk(assignment, 2, dim=-1) # [bs, num_node1, num_node2, neighbor_k, 1]
    index1 = index1.repeat(1, 1, 1, 1, neighbor1.size(-1)) # [bs, num_node1, num_node2, neighbor_k, dim]
    index2 = index2.repeat(1, 1, 1, 1, neighbor2.size(-1)) # [bs, num_node1, num_node2, neighbor_k, dim]
    neighbor1 = torch.gather(neighbor1, index=index1.long(), dim=-2)
    neighbor2 = torch.gather(neighbor2, index=index2.long(), dim=-2)

    return neighbor1, neighbor2


def modify_naive_similarity_para(naive_para):
    """Modify naive similarity model"""

    new_para = OrderedDict()
    for k in naive_para.keys():
        if 'naive' in k:
            new_k = k.replace('naive', 'graph')
        else:
            new_k = k
        new_para[new_k] = naive_para[k]
    return new_para


if __name__ == '__main__':

    a = torch.Tensor([[[1,2,3,4], [5,6,7,8], [9,10,11,12]], [[13,14,15,16], [17,18,19,20], [21,22,23,24]]]) #.randn(2,3,4)
    index = torch.Tensor([[0,2,1], [1,2,0]]).long() # [2, 3]
    index = index.unsqueeze(dim=-1).repeat(1,1,4) # [2, 3, 4]

    b = torch.gather(a, index=index, dim=1)
    print(a)
    print(b)

    a = 1
    pass