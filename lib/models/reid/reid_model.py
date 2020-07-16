import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch._thnn

from lib.models.backbone.googlenet import GoogLeNet


class ReIDModel(nn.Module):
    def __init__(self, n_parts=8, dim_per_part=64, im_info=None, pretrained=False, normalize=True):
        super(ReIDModel, self).__init__()
        self.n_parts = n_parts
        self.dim_per_part = dim_per_part

        self.im_info = im_info

        self.output_dim = n_parts * dim_per_part
        self.normalize = normalize

        self.feat_conv = GoogLeNet()
        self.conv_input_feat = nn.Conv2d(self.feat_conv.output_channels, 512, 1)

        # part net
        self.conv_att = nn.Conv2d(512, self.n_parts, 1)

        for i in range(self.n_parts):
            setattr(self, 'linear_feature{}'.format(i+1), nn.Linear(512, self.dim_per_part))

        if pretrained:
            model_path = 'model_weight/reid_model_googlenet.pth'
            weight = torch.load(model_path)
            self.load_state_dict(weight)

    def forward(self, x):
        feature = self.feat_conv(x)
        feature = self.conv_input_feat(feature)

        att_weights = torch.sigmoid(self.conv_att(feature))


        linear_feautres = []
        for i in range(self.n_parts):
            tmp_feature = feature * torch.unsqueeze(att_weights[:, i], 1)
            tmp_feature = F.avg_pool2d(tmp_feature, tmp_feature.size()[2:4])
            linear_feautres.append(
                getattr(self, 'linear_feature{}'.format(i + 1))(tmp_feature.view(tmp_feature.size(0), -1))
            )
        linear_feautres = torch.cat(linear_feautres, 1)
        if self.normalize:
            linear_feautres = linear_feautres / torch.clamp(torch.norm(linear_feautres, 2, 1, keepdim=True), min=1e-6)

        return linear_feautres
