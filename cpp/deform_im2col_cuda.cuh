#include <cstdio>
#include <algorithm>
#include <cstring>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// #include <THC/THC.h>
#include <THC/THCAtomics.cuh>
// #include <THC/THCDeviceUtils.cuh>

#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}


template <typename scalar_t>
__global__ void deformable_im2col_gpu_kernel(const int n,
                                              const scalar_t *data_im,
                                              const int batch_size, const int num_channels,
                                              const int height, const int width, 
                                              const int kernel_h, const int kernel_w,
                                              const int stride_h, const int stride_w,
                                              const int pad_h, const int pad_w,
                                              const int dilation_h, const int dilation_w,
                                              const int height_col, const int width_col,
                                              scalar_t *data_col)
{
  // launch channels * batch_size * height_col * width_col cores
  // const int num_kernels = channels * batch_size * height_col * width_col;
  CUDA_KERNEL_LOOP(index, n)
  {
    // NOTE(CharlesShang): different from Dai Jifeng's MXNet implementation, col_buffer is of shape (c*kw*kh, N, oh, ow)
    // here columns is of shape (N, c*kw*kh, oh * ow), need to adapt axis
    // NOTE(Jiarui XU): different from CharlesShang's implementation, col_buffer is of shape (N, c*kw*kh, oh * ow)
    // here columns is of shape (c*kw*kh, N, oh, ow), need to adapt axis

    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int b_col = (index / width_col / height_col) % batch_size;
    const int c_im = (index / width_col / height_col) / batch_size;
    const int c_col = c_im * kernel_h * kernel_w;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;

    scalar_t *data_col_ptr = data_col + ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    const int im_pos = (b_col * num_channels + c_im) * height * width;

    for (int i = 0; i < kernel_h; ++i)
    {
      for (int j = 0; j < kernel_w; ++j)
      {
        scalar_t val = static_cast<scalar_t>(0);
        const int h_im = h_in + i * dilation_h;
        const int w_im = w_in + j * dilation_w;
        if (h_im >=0 && w_im >=0 && h_im < height && w_im < width)
        {
          val = data_im[im_pos+h_im*width+w_im];
        }
        *data_col_ptr = val;
        data_col_ptr += batch_size * height_col * width_col;
      }
    }
  }
}

template <typename scalar_t>
__global__ void deformable_col2im_gpu_kernel(const int n,
                                              const scalar_t *data_col,
                                              const int batch_size, const int channels,
                                              const int height, const int width,
                                              const int kernel_h, const int kernel_w,
                                              const int stride_h, const int stride_w,
                                              const int pad_h, const int pad_w,
                                              const int dilation_h, const int dilation_w,
                                              const int height_col, const int width_col,
                                              scalar_t *grad_im)
{
  
  // launch kernels = channels * kernel_h * kernel_w * batch_size * height_col * width_col;
  CUDA_KERNEL_LOOP(index, n)
  {
    const int j = (index / width_col / height_col / batch_size) % kernel_w;
    const int i = (index / width_col / height_col / batch_size / kernel_w) % kernel_h;
    const int c = index / width_col / height_col / batch_size / kernel_w / kernel_h;
    
    // compute the start and end of the output
    int w_out = index % width_col;
    int h_out = (index / width_col) % height_col;
    int b = (index / width_col / height_col) % batch_size;
    int w_in = w_out * stride_w - pad_w;
    int h_in = h_out * stride_h - pad_h;

    const scalar_t cur_top_grad = data_col[index];
    const int cur_h = h_in + i * dilation_h;
    const int cur_w = w_in + j * dilation_w;
    if (cur_h >= 0 && cur_h < height && cur_w >= 0 && cur_w < width)
    {
      int cur_bottom_grad_pos = ((b * channels + c) * height + cur_h) * width + cur_w;
      atomicAdd(grad_im + cur_bottom_grad_pos, cur_top_grad);
    }
  }
}


template <typename scalar_t>
void deformable_im2col_cuda(cudaStream_t stream,
                              const scalar_t* data_im,
                              const int batch_size, const int channels, 
                              const int height_im, const int width_im, 
                              const int kernel_h, const int kernel_w,
                              const int stride_h, const int stride_w, 
                              const int pad_h, const int pad_w, 
                              const int dilation_h, const int dilation_w,
                              const int height_col, const int width_col, 
                              scalar_t* data_col) 
{
  const int num_kernels = channels * batch_size * height_col * width_col;
  deformable_im2col_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
      num_kernels, 
      data_im, 
      batch_size, channels, 
      height_im, width_im, 
      kernel_h, kernel_w,
      stride_h, stride_w, 
      pad_h, pad_w, 
      dilation_h, dilation_w,
      height_col, width_col, 
      data_col);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }
}

template <typename scalar_t>
void deformable_col2im_cuda(cudaStream_t stream,
                              const scalar_t* data_col,
                              const int batch_size, const int channels, 
                              const int height_im, const int width_im, 
                              const int kernel_h, const int kernel_w,
                              const int stride_h, const int stride_w, 
                              const int pad_h, const int pad_w, 
                              const int dilation_h, const int dilation_w, 
                              const int height_col, const int width_col, 
                              scalar_t* grad_im)
{
  const int num_kernels = channels * kernel_h * kernel_w * batch_size * height_col * width_col;
  deformable_col2im_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
        num_kernels, 
        data_col,
        batch_size, channels, 
        height_im, width_im,
        kernel_h, kernel_w, 
        stride_h, stride_w,
        pad_h, pad_h, 
        dilation_h, dilation_w,
        height_col, width_col, 
        grad_im);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in deformable_col2im_cuda: %s\n", cudaGetErrorString(err));
  }
}
