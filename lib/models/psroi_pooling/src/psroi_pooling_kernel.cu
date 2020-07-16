#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>


#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024 // the number of threads per block, i.e blockDim.x = 1024
inline int GET_NUM_BLOCKS(const int n)
{
    return (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}


template <typename scalar_t>
__global__ void PSROIPoolForward(
    const int nthreads,
    const scalar_t* bottom_data,
    const scalar_t spatial_scale,
    const int height,
    const int width,
    const int channels,
    const int pooled_height,
    const int pooled_width,
    const int group_size,
    const int output_dim,
    const scalar_t* bottom_rois,
    scalar_t* top_data,
    int* mapping_channel)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        // (n, c, ph, pw) is an element in the pooled output
        int pw = index % pooled_width;
      	int ph = (index / pooled_width) % pooled_height;
      	int ctop = (index / pooled_width / pooled_height) % output_dim;
      	int n = index / pooled_width / pooled_height / output_dim;

        bottom_rois += n * 5;
        int roi_batch_ind = bottom_rois[0];
	    scalar_t roi_start_w = static_cast<scalar_t>(round(bottom_rois[1])) * spatial_scale;
      	scalar_t roi_start_h = static_cast<scalar_t>(round(bottom_rois[2])) * spatial_scale;
      	scalar_t roi_end_w = static_cast<scalar_t>(round(bottom_rois[3]) + 1.) * spatial_scale;
      	scalar_t roi_end_h = static_cast<scalar_t>(round(bottom_rois[4]) + 1.) * spatial_scale;

        // Force malformed ROIs to be 1x1
        scalar_t roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0
      	scalar_t roi_height = max(roi_end_h - roi_start_h, 0.1);

        scalar_t bin_size_h = (scalar_t)(roi_height) / (scalar_t)(pooled_height);
        scalar_t bin_size_w = (scalar_t)(roi_width) / (scalar_t)(pooled_width);

        int hstart = floor(static_cast<scalar_t>(ph) * bin_size_h + roi_start_h);
      	int wstart = floor(static_cast<scalar_t>(pw)* bin_size_w + roi_start_w);
      	int hend = ceil(static_cast<scalar_t>(ph + 1) * bin_size_h + roi_start_h);
      	int wend = ceil(static_cast<scalar_t>(pw + 1) * bin_size_w + roi_start_w);

        // Add roi offsets and clip to input boundaries
        hstart = min(max(hstart, 0), height);
      	hend = min(max(hend, 0), height);
      	wstart = min(max(wstart, 0), width);
      	wend = min(max(wend, 0), width);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        int gw = pw;
      	int gh = ph;
      	int c = (ctop*group_size + gh)*group_size + gw;

        bottom_data += (roi_batch_ind * channels + c) * height * width;
        scalar_t out_sum = 0;
      	for (int h = hstart; h < hend; ++h) {
      	  for (int w = wstart; w < wend; ++w) {
      	    int bottom_index = h*width + w;
      	    out_sum += bottom_data[bottom_index];
      	  }
      	}
      	scalar_t bin_area = (hend - hstart)*(wend - wstart);
      	top_data[index] = is_empty? 0. : out_sum/bin_area;
      	mapping_channel[index] = c;
    }
}


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
    at::Tensor mapping_channel)
{
    //const int kThreadsPerBlock = 1024;

    const int output_size = output_dim * pooled_height * pooled_width * num_rois;
    const int num_blocks = GET_NUM_BLOCKS(output_size);

    AT_DISPATCH_FLOATING_TYPES(bottom_data.type(), "PSROIPoolForwardLauncher", ([&] {
        PSROIPoolForward<scalar_t><<<num_blocks, THREADS_PER_BLOCK>>>(
            output_size,
            bottom_data.data<scalar_t>(),
            spatial_scale,
            height,
            width,
            channels,
            pooled_height,
            pooled_width,
            group_size,
            output_dim,
            bottom_rois.data<scalar_t>(),
            top_data.data<scalar_t>(),
            mapping_channel.data<int>());
    }));
    return 1;
}


template <typename scalar_t>
__global__ void PSROIPoolBackward(
    const int nthreads,
    const scalar_t* top_diff,
    const int* mapping_channel,
    const int num_rois,
    const scalar_t spatial_scale,
    const int height,
    const int width,
    const int channels,
    const int pooled_height,
    const int pooled_width,
    const int output_dim,
    scalar_t* bottom_diff,
    const scalar_t* bottom_rois) {
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {

      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int n = index / pooled_width / pooled_height / output_dim;

      // [start, end) interval for spatial sampling
      bottom_rois += n * 5;
      int roi_batch_ind = bottom_rois[0];
      scalar_t roi_start_w = static_cast<scalar_t>(round(bottom_rois[1])) * spatial_scale;
      scalar_t roi_start_h = static_cast<scalar_t>(round(bottom_rois[2])) * spatial_scale;
      scalar_t roi_end_w = static_cast<scalar_t>(round(bottom_rois[3]) + 1.) * spatial_scale;
      scalar_t roi_end_h = static_cast<scalar_t>(round(bottom_rois[4]) + 1.) * spatial_scale;

      // Force too small ROIs to be 1x1
      scalar_t roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0
      scalar_t roi_height = max(roi_end_h - roi_start_h, 0.1);

      // Compute w and h at bottom
      scalar_t bin_size_h = roi_height / static_cast<scalar_t>(pooled_height);
      scalar_t bin_size_w = roi_width / static_cast<scalar_t>(pooled_width);

      int hstart = floor(static_cast<scalar_t>(ph)* bin_size_h + roi_start_h);
      int wstart = floor(static_cast<scalar_t>(pw)* bin_size_w + roi_start_w);
      int hend = ceil(static_cast<scalar_t>(ph + 1) * bin_size_h + roi_start_h);
      int wend = ceil(static_cast<scalar_t>(pw + 1) * bin_size_w + roi_start_w);
      // Add roi offsets and clip to input boundaries
      hstart = min(max(hstart, 0), height);
      hend = min(max(hend, 0), height);
      wstart = min(max(wstart, 0), width);
      wend = min(max(wend, 0), width);
      bool is_empty = (hend <= hstart) || (wend <= wstart);

      // Compute c at bottom
      int c = mapping_channel[index];
      scalar_t* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;
      scalar_t bin_area = (hend - hstart)*(wend - wstart);
      scalar_t diff_val = is_empty ? 0. : top_diff[index] / bin_area;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          int bottom_index = h*width + w;
          //caffe_gpu_atomic_add(diff_val, offset_bottom_diff + bottom_index);
          atomicAdd(offset_bottom_diff + bottom_index, diff_val);
        }
      }
  }
}

int PSROIPoolBackwardLauncher(
    at::Tensor top_diff,
    /*const int**/ at::Tensor mapping_channel,
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
    at::Tensor bottom_rois)
{
    //const int output_size = output_dim * height * width * channels;
    const int output_size = output_dim * pooled_height * pooled_width * num_rois;
    const int num_blocks = GET_NUM_BLOCKS(output_size);

    AT_DISPATCH_FLOATING_TYPES(top_diff.type(), "PSROIPoolBackwardLauncher", ([&] {
        PSROIPoolBackward<scalar_t><<<num_blocks, THREADS_PER_BLOCK>>>(
            output_size,
            top_diff.data<scalar_t>(),
            mapping_channel.data<int>(),
            num_rois,
            spatial_scale,
            height,
            width,
            channels,
            pooled_height,
            pooled_width,
            output_dim,
            bottom_diff.data<scalar_t>(),
            bottom_rois.data<scalar_t>());
    }));

    return 1;
}

