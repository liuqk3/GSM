#include <torch/torch.h>
//#include <torch/extension.h>
#include <math.h>

// cuda function declaration
int PSROIPoolForwardLauncher(
    at::Tensor bottom_data,
    const float spatial_scale,
    const int num_rois,
    const int height,
    const int width,
    const int channels,
    const int pooled_height,
    const int pooled_width,
    at::Tensor bottom_rois,
    const int group_size,
    const int output_dim,
    at::Tensor top_data,
    at::Tensor mapping_channel);


int PSROIPoolBackwardLauncher(
    at::Tensor top_diff,
    at::Tensor mapping_channel,
    const int batch_size,
    const int num_rois,
    const float spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_width,
    const int pooled_height,
    const int output_dim,
    at::Tensor bottom_diff,
    at::Tensor bottom_rois);


// C++ interface
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_TENSOR(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int psroi_pooling_forward_cuda(
    int pooled_height,
    int pooled_width,
    float spatial_scale,
    int group_size,
    int output_dim,
    at::Tensor features,
    at::Tensor rois,
    at::Tensor output,
    at::Tensor mappingchannel)
{

    CHECK_TENSOR(features);
    CHECK_TENSOR(rois);
    CHECK_TENSOR(output);
    CHECK_TENSOR(mappingchannel);

	//Get # of Rois
	int num_rois = rois.size(0);
	int size_rois = rois.size(1);
	if (size_rois!=5)
	{
		return -1;
	}

	//Get # of batch_size
	int batch_size = features.size(0);
	int data_height = features.size(2);
	int data_width = features.size(3);
	int num_channels = features.size(1);

	// call the gpu kernel for psroi_pooling
	PSROIPoolForwardLauncher(
	    features,
	    spatial_scale,
	    num_rois,
	    data_height,
	    data_width,
	    num_channels,
	    pooled_height,
	    pooled_width,
	    rois,
	    group_size,
	    output_dim,
	    output,
	    mappingchannel);
	return 1;
}


int psroi_pooling_backward_cuda(
    int pooled_height,
    int pooled_width,
    float spatial_scale,
    int output_dim,
    at::Tensor top_grad,
    at::Tensor rois,
    at::Tensor bottom_grad,
    at::Tensor mappingchannel)
{
    CHECK_TENSOR(top_grad);
    CHECK_TENSOR(rois);
    CHECK_TENSOR(bottom_grad);
    CHECK_TENSOR(mappingchannel);

    // Number of ROIs
    int num_rois = rois.size(0);
    int size_rois = rois.size(1);
    if (size_rois != 5)
    {
        return -1;
    }
    // batch size
    int batch_size = bottom_grad.size(0);

    // data height
    int data_height = bottom_grad.size(2);
    // data width
    int data_width = bottom_grad.size(3);
    // Number of channels
    int num_channels = bottom_grad.size(1);

    PSROIPoolBackwardLauncher(
        top_grad,
        mappingchannel,
        batch_size,
        num_rois,
        spatial_scale,
        num_channels,
        data_height,
        data_width,
        pooled_width,
        pooled_height,
        output_dim,
        bottom_grad,
        rois);
    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("psroi_pooling_forward_cuda", &psroi_pooling_forward_cuda, "psroi_pooling_forward_cuda (CUDA)");
  m.def("psroi_pooling_backward_cuda", &psroi_pooling_backward_cuda, "psroi_pooling_backward_cuda (CUDA)");
}