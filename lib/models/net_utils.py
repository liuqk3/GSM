
import torch
from lib.models.reid.reid_model import ReIDModel
from collections import OrderedDict


def model_is_cuda(model):
    p = next(model.parameters())
    return p.is_cuda


def get_model_device(model):
    if model_is_cuda(model):
        p = next(model.parameters())
        return p.get_device()
    else:
        return None


def load_model_para(model_para, net):
    """load model parameters
    Args:
        model_para: the path to model
        net: the network model
    """

    if hasattr(net, 'module'):
        model_para_tmp = OrderedDict()
        for key in model_para.keys():
            if 'module.' not in key:
                key_tmp = 'module.' + key
            else:
                key_tmp = key
            model_para_tmp[key_tmp] = model_para[key]
    else:
        model_para_tmp = model_para
    net.load_state_dict(model_para_tmp)


def load_reid_model(model_cfg, model_path=None):
    model = ReIDModel(**model_cfg['init_args'])

    if model_path is None:
        model_path = 'model_weight/reid_model_googlenet.pth'
    model.load_state_dict(torch.load(model_path))
    print('Load ReID model from {}'.format(model_path))

    return model


if __name__ == '__main__':
    eye = torch.eye(5)
    print(eye)