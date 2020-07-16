import os
import torch
from lib.models.config.config import load_config as load_model_config
from lib.models.graph.similarity_model import GraphSimilarityl
from lib.tracking.association_model import AssociationModel
from lib.models.detection.frcnn_fpn import FRCNN_FPN
from lib.tracking.detection_model import DetectionModel
from lib.models.classification.classifier import PatchClassifier


# =================================================================== #
#                    OnlineTracker config                             #
# =================================================================== #
TrackerConfig = {
    'init_args':
        {
            # 'min_cls_score': 0.4,  # the score of each box is re-estimated
            'min_asso_dist':0.85, # 0.6 for naive, # 0.85 for graph in ijcai, # the min dist between tracks and detections, greater dist will not be associated
            'max_time_lost': 30,  # how long a track will remain in lost before it is removed
            'association_model': None,  # the model used for tracking
            'match_type': 'graph_match',  # 'naive_math', 'graph_match'

            'detection_model': 'model_weight/frcnn_fpn_model_epoch_27.model',  # or None, the detector copied from Tracktor
            'classifier': None, #'model_weight/patch_classifier_squeezenet.pth', # the model to get the box classification score, copied from MOTDT, not used in our paper
            'use_kalman_filter': False, 
            'use_flow': True,
            'use_iou_match': False, # if False, the 
            'gate_cost_matrix': False, # whether use kalman filter to gate the cost matrix while performing association, only effective when a kalman filter is used
            'use_tracking': False, # can be set to perform SOT for each tracklet
            'use_refind': False, # whether to refind the lost tracks
            'use_neighbor': True, # whether to use neighbors when perform graph_match, only effective in 'graph_match'

            'use_reactive': True, # reactivate a lost tracklet ranther assign a new ID
            'strict_match': True,  # if true, all tracks (detections) will also be matched with padded detection (track)
            'debug': False,
        },
    'graph_model_name': 'GraphSimilarity_v5',
    'graph_model_path': 'model_weight/GraphSimilarity_v5_2_22.pth',
    'year': 'MOT16',
    'phase': 'val', # 'train', 'test', 'val'
    'det_root': None, # the directory that contain detection results, such as '/home/liuqk/Dataset/MOT/MOT16/POI_MOT16_det_feat/',
    'plot_type': 'none', # 'tracks', 'neighbors', 'none'
    'anchor_id_to_show': 0, # plot the neighbors of the objects with the ids
    'save_image': False,
    'track_result_path': None, # the directory to save the results, if None, the results will be saved in default directory
    'cuda': True,
}


def get_config(cfg=None):
    if cfg is None:
        cfg = TrackerConfig.copy()

    # get graph similarity model
    similar_model_config = os.path.join(os.path.dirname(cfg['graph_model_path']), 'config.json')
    similar_model_config = load_model_config(similar_model_config)
    similar_model_config = similar_model_config['GraphSimilarity']
    model = GraphSimilarityl(**similar_model_config['init_args'])

    model_para = torch.load(cfg['graph_model_path'], map_location=torch.device('cpu'))
    model_para = model_para['model']

    model.load_state_dict(model_para)
    print('Load model from {}'.format(cfg['graph_model_path']))
    if cfg['cuda']:
        model = model.cuda()
    model.eval()

    asso_model = AssociationModel(model=model) # , debug=cfg['init_args']['debug'])
    cfg['init_args']['association_model'] = asso_model

    # get the classifier
    assert cfg['init_args']['classifier'] is None or isinstance(cfg['init_args']['classifier'], str)
    if cfg['init_args']['classifier'] is not None:
        classifier = PatchClassifier(model_path=cfg['init_args']['classifier'], cuda=cfg['cuda'])
        cfg['init_args']['classifier'] = classifier
    else:
        cfg['init_args']['classifier'] = None

    # get the detector
    assert cfg['init_args']['detection_model'] is None or isinstance(cfg['init_args']['detection_model'], str)
    if cfg['init_args']['detection_model'] is not None:
        detector = FRCNN_FPN(num_classes=2)
        detector.load_state_dict(torch.load(cfg['init_args']['detection_model'], map_location=torch.device('cpu')))
        if cfg['cuda']:
            detector = detector.cuda()
        detection_model = DetectionModel(model=detector, debug=cfg['init_args']['debug'])
        cfg['init_args']['detection_model'] = detection_model
    else:
        cfg['init_args']['detection_model'] = None

    # prepare result path
    track_result_path, all_metric_file_path = get_results_path(cfg)

    if cfg['track_result_path'] is None:
        cfg['track_result_path'] = track_result_path
    cfg['all_metric_path'] = all_metric_file_path

    assert cfg['plot_type'] in ['tracks', 'neighbors', 'none']
    if cfg['save_image']:
        assert cfg['plot_type'] in ['tracks', 'neighbors']

    return cfg


def get_results_path(cfg):
    # prepare result path
    prefix = ''

    if cfg['init_args']['detection_model'] is not None:
        prefix += 'det'
    else:
        prefix += 'nodet'

    if cfg['init_args']['classifier'] is not None:
        prefix += '_cls'
    else:
        prefix += '_nocls'

    if cfg['init_args']['use_kalman_filter']:
        prefix += '_kf'
        if not cfg['init_args']['gate_cost_matrix']:
            raise ValueError('Kalman filter is used, but gate_cost_matrix is flase!')
    else:
        prefix += '_nokf'

    if cfg['init_args']['use_flow']:
        prefix += '_flow'
    else:
        prefix += '_noflow'

    if cfg['init_args']['use_flow'] and cfg['init_args']['use_kalman_filter']:
        raise ValueError('only one of kalman or flow can be used.')

    if cfg['init_args']['use_iou_match']:
        prefix += '_iou'
    else:
        prefix += '_noiou'

    if cfg['init_args']['gate_cost_matrix']:
        prefix = prefix + '_gate'
    else:
        prefix = prefix + '_nogate'

    if cfg['init_args']['use_tracking']:
        prefix += '_track'
    else:
        prefix += '_notrack'

    if cfg['init_args']['use_refind']:
        prefix += '_refind'
    else:
        prefix += '_norefind'

    match_type = cfg['init_args']['match_type']
    if match_type == 'graph_match':
        if cfg['init_args']['use_neighbor']:
            match_type = match_type + '_nei'
        else:
            match_type = match_type + '_no_nei'

    det_type = 'public'
    if cfg['det_root'] is not None:
        if 'POI' in cfg['det_root']:
            det_type = 'poi'
        elif 'Tracktor' in cfg['det_root'] or 'tracktor' in cfg['det_root']:
            det_type = 'tracktor'

    model_base_name = os.path.basename(cfg['graph_model_path']).split('.')[0]
    model_base_name = model_base_name.split('_') # such as ['GraphSimilarity, 2, 12]

    track_result_path = os.path.join('track_results', prefix, det_type,
                                     cfg['year'], cfg['phase'],  match_type,
                                     '_'.join(model_base_name[0:-1]), str(cfg['init_args']['min_asso_dist']),
                                     model_base_name[-1])

    all_metric_path = os.path.join('track_results', prefix, det_type,
                                   cfg['year'], cfg['phase'], match_type,
                                   '_'.join(model_base_name[0:-1]), str(cfg['init_args']['min_asso_dist']),
                                   'all_metric.txt')

    print('Track results will be saved in {}'.format(track_result_path))
    return track_result_path, all_metric_path
