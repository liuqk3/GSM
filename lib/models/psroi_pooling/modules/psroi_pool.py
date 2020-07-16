from torch.nn.modules.module import Module
from ..functions.psroi_pooling import PSRoIPoolingFunction, PSRoIPoolingFunction_v2


class PSRoIPool(Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale, group_size, output_dim):
        super(PSRoIPool, self).__init__()

        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        self.group_size = int(group_size)
        self.output_dim = int(output_dim)

    def forward(self, features, rois):
        # return PSRoIPoolingFunction(self.pooled_height, self.pooled_width, self.spatial_scale, self.group_size,
        #                             self.output_dim)(features, rois)
        return PSRoIPoolingFunction_v2.apply(features, rois,
                                             self.pooled_height, self.pooled_width,
                                             self.spatial_scale, self.group_size, self.output_dim)
