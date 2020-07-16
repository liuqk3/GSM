import os
import json
from lib.utils.io import save_json

config = {
            'description': "",

            # ====================================================================
            #                        backbone config                             #
            # ===================================================================#

            'ReIDModel': # the config for reid model should not be changed
                {
                    'init_args': {
                        'n_parts': 8,
                        'dim_per_part': 64,
                        'pretrained': True,
                        'normalize': True,

                        'im_info':{
                            'size': (80, 160),  # w, h
                            'scale': [1, 1, 1],
                            'mean': [104, 117, 123],
                            'var': [1, 1, 1],
                            'channel':'BRG',
                        },
                    },

                    'distance_metric': 'euclidean',
                },

            'ResNetBackbone': # the config for resnet
                {
                    'init_args':{
                        'layer': 34, #50, #
                        'fix_block': 1, # used to fix the block of ResNet
                        'output_dim': 256, # the number of dims of each input node (person) feature
                        'pretrained': True,

                        'im_info':{
                            'size': (64, 128),  # w, h
                            'scale': [255, 255, 255],
                            'mean': [0.406, 0.456, 0.485],
                            'var': [0.225, 0.224, 0.229],
                            'channel': 'RGB',
                        },
                    },

                },

            # =================================================================== #
            #                        GraphSimilarity config                         #
            # =================================================================== #
            'NaiveMatch':
                {
                    'init_args':
                        {
                            'n_in': 256,  # dim of features output by backbone
                            'loss_type': 'binary_cross_entropy',
                            'np_ratio': 2, # the ratio between the number of negative and positive samples
                            'do_drop': 0, # dropout probability,
                            'use_pos': False, # whether to use the position in final classification
                            'encode_pos': True,
                            'embed_pos': True,
                            'pos_dim': 64,  # the number of dimensions to embed pisition information
                            'wave_length': 1000,  # the wave length used to embed position information
                        }
                },
            'GraphMatch_v5':
            {
                'init_args':
                    {
                        'n_in': 256,  # dim of features output by backbone

                        'neighbor_k': 5,  # the number of neighbors used to match
                        'neighbor_type': 'pos', # how to get the neighbors, can be one of [learn_app_pos, learn_app, learn_pos, pos]
                                            # learn_app_pos: learn from appearance and position embeddings
                                            # learn_app: learn from appearance features
                                            # learn_pos: learn from position embeddings
                                            # pos: purely based on position, default is the euclidean distance between the center coordinates
                        'neighbor_weight_branch': 'none', # 'none', # whether to get the weights for neighbors, if not, we set the weight to 1
                        'absorb_weight': 0.75,  # the weight to absorb information from neighbors

                        'use_pos': True, # whether to use the position in the final classiication. If true, the weights of neighbors are also computed based on position features
                        'encode_pos': True,  # whether to encode the boxes using the function in Transformer
                        'embed_pos': True,  # whether to embed the position information by some layer
                        'pos_quantify': -1, # whether to quantify the position of objects
                        'pos_dim': 64, # the number of dimensions to encode position information, only effective when encode_pos is True
                        'wave_length': 1000,  # the wave length used to encode position information, only effective when encode_pos is True
                        'pos_dim_out': 256,  # the dimension of position feature after embeding

                        'do_drop': 0.,  # dropout probability
                        'loss_type': 'binary_cross_entropy',
                        'np_ratio': -1,  # the ratio between the number of negative and positive samples
                        'train_part': 'all',
                },
            },

            'GraphSimilarity':
                {
                    'init_args':{
                        'backbone_name': 'ResNetBackbone', # 'ReIDModel',
                        'match_name': "GraphMatch_v5", #'GraphMatch', 'NaiveMatch', 'NaiveMatch,GraphMatch'
                        'backbone_args': None, #
                        'graphmatch_args': None,
                        'naivematch_args': None,
                        'pad_boxes': False, # whether the inputed data is padded a track box and a det box
                        'train_part': 'backbone, graph_match, naive_match', # 'backbone, graph_match, naive_match',
                    },
                },

            # =================================================================== #
            #                        Dataset config                               #
            # =================================================================== #
            'MOTTrackletStack':
                {
                    'year': 'MOT17',
                    'min_num_frame': 2,
                    'max_num_frame':10,
                    'num_frame': -1,
                    'max_num_node': 40,
                    'min_num_node': 6,
                    'pad_boxes': False, # whether to pad a track box and a detection box
                    'im_info': None,
                    'cache_dir': 'data_cache',
                    'augment': True, # whether to augment the images
            },
            'MOTFramePair':
                {
                    'year': 'MOT17',
                    'min_num_frame': 2,
                    'max_num_frame': 10,
                    'num_frame': 2,
                    'max_num_node': -1,
                    'min_num_node': -1,
                    'pad_boxes': False, # whether to pad a track box and a detection box
                    'im_info': None,
                    'cache_dir': 'data_cache',
                    'augment': False, # whether to augment the images
                },
        }


def get_config():
    """Load config, we need to do some post process"""
    cfg = config.copy()

    # GraphSimilarity
    cfg['GraphSimilarity']['init_args']['backbone_args'] = cfg[cfg['GraphSimilarity']['init_args']['backbone_name']]['init_args']
    cfg['GraphSimilarity']['init_args']['pad_boxes'] = cfg['MOTFramePair']['pad_boxes']
    match_names = cfg['GraphSimilarity']['init_args']['match_name']
    match_names = match_names.strip().split(',')
    for n in match_names:
        match_args = cfg[n]['init_args']
        n_tmp = n.split('_')[0] # GraphMatch_v2 -> GraphMatch
        cfg['GraphSimilarity']['init_args'][n_tmp.lower()+'_args'] = match_args

    # Dataset
    cfg['MOTTrackletStack']['im_info'] = cfg[cfg['GraphSimilarity']['init_args']['backbone_name']]['init_args']['im_info']
    cfg['MOTFramePair']['im_info'] = cfg[cfg['GraphSimilarity']['init_args']['backbone_name']]['init_args']['im_info']

    return cfg


def merge_arg_to_config(args, cfg):

    args_dict = {}
    for k, v in vars(args).items():
        args_dict[k] = v

    cfg['args'] = args_dict
    return cfg


def save_config(cfg, cfg_path):
    """Save config to a json file, so the config can be reused"""
    save_json(in_dict=cfg, save_path=cfg_path)


def load_config(cfg_path):
    """Load the config from a json file"""
    if not os.path.exists(cfg_path):
        raise RuntimeError('file {} does not exists!'.format(cfg_path))

    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    return cfg


