
import torch
from lib.models import net_utils
from lib.datasets.dataset_utils import extract_image_patches


def extract_reid_features(reid_model, image, tlbrs):
    # image: bgr
    if len(tlbrs) == 0:
        return torch.FloatTensor()

    patches = extract_image_patches(image, tlbrs, patch_size=reid_model.im_info['size'],
                                    scale=reid_model.im_info['scale'], mean=reid_model.im_info['mean'],
                                    var=reid_model.im_info['var'], channel=reid_model.im_info['channel'])

    gpu = net_utils.get_model_device(reid_model)

    im_var = torch.Tensor(patches)
    with torch.no_grad():
        if gpu is not None:
            im_var = im_var.cuda(gpu)
        features = reid_model(im_var)

    return features
