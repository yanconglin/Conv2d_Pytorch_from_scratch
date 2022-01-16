#include <vector>
#include "deform_im2col_cuda.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// #include <THC/THC.h>
// #include <THC/THCAtomics.cuh>
// #include <THC/THCDeviceUtils.cuh>

// extern THCState *state;

// author: Charles Shang
// https://github.com/torch/cunn/blob/master/lib/THCUNN/generic/SpatialConvolutionMM.cu


at::Tensor
deform_conv_cuda_forward(const at::Tensor &input,
                    const at::Tensor &weight,
                    const at::Tensor &bias,
                    const int kernel_h,
                    const int kernel_w,
                    const int stride_h,
                    const int stride_w,
                    const int pad_h,
                    const int pad_w,
                    const int dilation_h,
                    const int dilation_w,
                    const int groups,
                    const int im2col_step)
{
    // THCAssertSameGPU(THCudaTensor_checkGPU(state, 5, input, weight, bias, offset, mask));

    AT_ASSERTM(input.is_contiguous(), "input tensor has to be contiguous");
    AT_ASSERTM(weight.is_contiguous(), "weight tensor has to be contiguous");
    AT_ASSERTM(bias.is_contiguous(), "bias tensor has to be contiguous");

    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(weight.type().is_cuda(), "weight must be a CUDA tensor");
    AT_ASSERTM(bias.type().is_cuda(), "bias must be a CUDA tensor");

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int channels_out = weight.size(0);
    const int channels_kernel = weight.size(1);
    const int kernel_h_ = weight.size(2);
    const int kernel_w_ = weight.size(3);

    const int im2col_step_ = std::min(batch, im2col_step);

    AT_ASSERTM(batch % im2col_step_ == 0, "batch(%d) must divide im2col_step(%d)", batch, im2col_step_);

    AT_ASSERTM((channels % groups == 0) && (channels_out % groups == 0), 
        "channels(%d) and channels_out(%d) must divide groups(%d)", channels, channels_out, groups);

    // printf("Kernels: %d %d %d %d\n", kernel_h_, kernel_w_, kernel_w, kernel_h);
    // printf("Channels: %d %d\n", channels, channels_kernel);
    // printf("Channels: %d %d\n", channels_out, channels_kernel);

    AT_ASSERTM(kernel_h_ == kernel_h && kernel_w_ == kernel_w,
               "Input shape and kernel shape wont match: (%d x %d vs %d x %d).", kernel_h_, kernel_w, kernel_h_, kernel_w_);

    AT_ASSERTM(channels == (channels_kernel * groups),
               "Input shape and kernel channels wont match: (%d vs %d).", channels, channels_kernel * groups);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    auto output = at::empty({batch * height_out * width_out, channels_out}, input.options());

    // prepare groups weight and bias
    auto weight_g = weight.view({groups, channels_out/groups, channels_kernel, kernel_h, kernel_w});
    auto bias_g = bias.view({groups, channels_out/groups});

    // define alias for easy use
    const int batch_n = im2col_step_;
    const int per_input_size = channels * height * width;
    auto output_n = output.view({batch/im2col_step_, batch_n * height_out * width_out, channels_out});
    for (int n = 0; n < batch/im2col_step_; ++n)
    {
        auto columns = at::empty({channels * kernel_h * kernel_w, batch_n * height_out * width_out}, input.options());
        AT_DISPATCH_FLOATING_TYPES(input.type(), "deform_conv_forward_cuda", ([&] {
            deformable_im2col_cuda(at::cuda::getCurrentCUDAStream(),
                                             input.data<scalar_t>() + n * im2col_step_ * per_input_size,
                                             batch_n, channels, 
                                             height, width,
                                             kernel_h, kernel_w,
                                             stride_h, stride_w, 
                                             pad_h, pad_w, 
                                             dilation_h, dilation_w,
                                             height_out, width_out, 
                                             columns.data<scalar_t>());

        }));

        // auto columns_m = columns.t();
        // auto weight_m = weight.view({channels_out, channels_kernel * kernel_h * kernel_w}).t();
        // output = at::addmm(bias, columns_m, weight_m);
        auto columns_g = columns.view({groups, channels/groups * kernel_h * kernel_w, batch_n * height_out * width_out});
        auto output_g = output_n.select(0, n).view({batch_n * height_out * width_out, groups, channels_out/groups});
        for (int g = 0; g < groups; ++g)
        {
            auto columns_gm = columns_g.select(0, g).t();
            auto weight_gm = weight_g.select(0, g).view({channels_out/groups, channels_kernel * kernel_h * kernel_w}).t();
            auto output_m = at::addmm(bias_g.select(0, g), columns_gm, weight_gm);
            output_g.select(1, g) = output_m.view({batch_n * height_out * width_out, channels_out/groups});
        }

    }

    output = output.view({batch, height_out, width_out, channels_out}).permute({0, 3, 1, 2}).contiguous();

    return output;
}

std::vector<at::Tensor> deform_conv_cuda_backward(const at::Tensor &input,
                                             const at::Tensor &weight,
                                             const at::Tensor &bias,
                                             const at::Tensor &grad_output,
                                             const int kernel_h, 
                                             const int kernel_w,
                                             const int stride_h, 
                                             const int stride_w,
                                             const int pad_h, 
                                             const int pad_w,
                                             const int dilation_h, 
                                             const int dilation_w,
                                             const int groups,
                                             const int im2col_step)
{

    AT_ASSERTM(input.is_contiguous(), "input tensor has to be contiguous");
    AT_ASSERTM(weight.is_contiguous(), "weight tensor has to be contiguous");
    AT_ASSERTM(bias.is_contiguous(), "weight tensor has to be contiguous");

    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(weight.type().is_cuda(), "weight must be a CUDA tensor");
    AT_ASSERTM(bias.type().is_cuda(), "bias must be a CUDA tensor");

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int channels_out = weight.size(0);
    const int channels_kernel = weight.size(1);
    const int kernel_h_ = weight.size(2);
    const int kernel_w_ = weight.size(3);

    const int batch_ = grad_output.size(0);
    const int channels_out_ = grad_output.size(1);
    const int height_out_ = grad_output.size(2);
    const int width_out_ = grad_output.size(3);

    const int im2col_step_ = std::min(im2col_step, batch);

    AT_ASSERTM(batch % im2col_step_ == 0, "batch(%d) must divide im2col_step(%d)", batch, im2col_step_);

    AT_ASSERTM((channels % groups == 0) && (channels_out % groups == 0), 
        "channels(%d) and channels_out(%d) must divide groups(%d)", channels, channels_out, groups);

    AT_ASSERTM(kernel_h_ == kernel_h && kernel_w_ == kernel_w,
               "Input shape and kernel shape wont match: (%d x %d vs %d x %d).", kernel_h_, kernel_w, kernel_h_, kernel_w_);

    AT_ASSERTM(channels == (channels_kernel * groups),
               "Input shape and kernel channels wont match: (%d vs %d).", channels, channels_kernel * groups);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    AT_ASSERTM(batch == batch_,
               "Input shape and grad_out batch wont match: (%d vs %d).", batch, batch_);

    AT_ASSERTM(channels_out == channels_out_,
               "Input shape and grad_out channels_out wont match: (%d vs %d).", channels_out, channels_out_);

    AT_ASSERTM(height_out == height_out_ && width_out == width_out_,
               "Input shape and grad_out shape wont match: (%d x %d vs %d x %d).", height_out, height_out_, width_out, width_out_);

    auto grad_input = at::zeros_like(input);
    auto grad_weight = at::zeros_like(weight);
    auto grad_bias = at::zeros_like(bias);

    // auto grad_output_m = grad_output.permute({1, 0, 2, 3}).contiguous().view({channels_out, batch * height_out * width_out});
    // auto weight_m = weight.view({channels_out, channels_kernel * kernel_h * kernel_w}).t();
    // columns = at::mm(weight_m, grad_output_m);

    // prepare groups weight and bias
    auto weight_g = weight.view({groups, channels_out/groups, channels_kernel, kernel_h, kernel_w});
    auto grad_weight_g = grad_weight.view({groups, channels_out/groups, channels_kernel, kernel_h, kernel_w});
    auto grad_bias_g = grad_bias.view({groups, channels_out/groups});

    const int batch_n = im2col_step_;
    const int per_input_size = channels * height * width;
    auto grad_output_n = grad_output.view({batch/im2col_step_, batch_n, channels_out, height_out, width_out});
    for (int n = 0; n < batch/im2col_step_; ++n)
    {
        auto ones = at::ones({batch_n * height_out * width_out}, input.options());
        auto grad_output_g = grad_output_n.select(0, n).view({batch_n, groups, channels_out/groups, height_out, width_out});
        auto columns = at::empty({channels * kernel_h * kernel_w, batch_n * 1 * height_out * width_out}, input.options());
        auto columns_g = columns.view({groups, channels/groups * kernel_h * kernel_w, batch_n * height_out * width_out});
        for (int g = 0; g < groups; ++g)
        {
            auto grad_output_gm = grad_output_g.select(1, g).permute({1, 0, 2, 3}).contiguous().view({channels_out/groups, batch_n * height_out * width_out});
            auto weight_gm = weight_g.select(0, g).view({channels_out/groups, channels_kernel * kernel_h * kernel_w}).t();
            columns_g.select(0, g) = at::mm(weight_gm, grad_output_gm);
        }

        AT_DISPATCH_FLOATING_TYPES(input.type(), "deform_conv_backward_cuda", ([&] {
            // gradient w.r.t. input data
            deformable_col2im_cuda(at::cuda::getCurrentCUDAStream(),
                                             columns.data<scalar_t>(),
                                             batch_n, channels, 
                                             height, width,
                                             kernel_h, kernel_w,
                                             stride_h, stride_w,
                                             pad_h, pad_w, 
                                             dilation_h, dilation_w,
                                             height_out, width_out, 
                                             grad_input.data<scalar_t>() + n * im2col_step_ * per_input_size);

            // gradient w.r.t. weight, dWeight should accumulate across the batch and groups
            deformable_im2col_cuda(at::cuda::getCurrentCUDAStream(),
                                             input.data<scalar_t>() + n * im2col_step_ * per_input_size,
                                             batch_n, channels, 
                                             height, width,
                                             kernel_h, kernel_w,
                                             stride_h, stride_w,
                                             pad_h, pad_w, 
                                             dilation_h, dilation_w,
                                             height_out, width_out, 
                                             columns.data<scalar_t>());

        }));

        // auto grad_output_m = grad_output.permute({1, 0, 2, 3}).contiguous().view({channels_out, batch * height_out * width_out});
        // grad_weight = at::mm(grad_output_m, columns.t()).view_as(weight);
        // grad_bias = at::mv(grad_output_m, ones);
        // auto grad_output_g = grad_output.view({batch, groups, channels_out/groups, height_out, width_out});
        // auto columns_g = columns.view({groups, channels/groups * kernel_h * kernel_w, batch * height_out * width_out});
        for (int g = 0; g < groups; ++g)
        {
            auto grad_output_gm = grad_output_g.select(1, g).permute({1, 0, 2, 3}).contiguous().view({channels_out/groups, batch_n * height_out * width_out});
            auto columns_gm = columns_g.select(0, g).t();
            auto grad_weight_gm = grad_weight_g.select(0, g).view({channels_out/groups, channels_kernel * kernel_h * kernel_w});
            auto grad_bias_gm = grad_bias_g.select(0, g);
            grad_weight_g.select(0, g) = at::addmm(grad_weight_gm, grad_output_gm, columns_gm).view_as(grad_weight_g.select(0, g));
            grad_bias_g.select(0, g) = at::addmv(grad_bias_gm, grad_output_gm, ones);
        }

    }

    return {
        grad_input, grad_weight, grad_bias
    };
}
