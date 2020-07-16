from torch.utils.cpp_extension import load
import os

cur_path = os.path.dirname(os.path.abspath(__file__))

build_dir = os.path.join(cur_path, '_ext')
if not os.path.exists(build_dir):
    os.mkdir(build_dir)

sources = ['src/psroi_pooling_cuda.cpp', 'src/psroi_pooling_kernel.cu']
sources = [os.path.join(cur_path, fname) for fname in sources]

# see: https://stackoverflow.com/questions/37566987/cuda-atomicadd-for-doubles-definition-error
extra_cuda_cflags = ['-arch=sm_60']

cuda_building = load(name='psroi_pooling',
                 sources=sources,
                 extra_cuda_cflags=extra_cuda_cflags,
                 build_directory=build_dir,
                 verbose=True)
# help(cuda_building)