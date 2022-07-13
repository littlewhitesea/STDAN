#pragma once

#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

void modulated_deform_conv_cuda_forward(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor ones,
    at::Tensor offset,
    at::Tensor mask,
    at::Tensor output,
    at::Tensor columns,
    int kernel_h,
    int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const int group,
    const int deformable_group,
    const bool with_bias);

void modulated_deform_conv_cuda_backward(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor ones,
    at::Tensor offset,
    at::Tensor mask,
    at::Tensor columns,
    at::Tensor grad_input,
    at::Tensor grad_weight,
    at::Tensor grad_bias,
    at::Tensor grad_offset,
    at::Tensor grad_mask,
    at::Tensor grad_output,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int group,
    int deformable_group,
    const bool with_bias);



at::Tensor
dsp_v2_forward(const at::Tensor &input,
               const at::Tensor &offset,
               const at::Tensor &mask,
               const int kernel_h,
               const int kernel_w,
               const int stride_h,
               const int stride_w,
               const int pad_h,
               const int pad_w,
               const int dilation_h,
               const int dilation_w,
               const int deformable_group)
{
    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return dsp_v2_cuda_forward(input, offset, mask,
                                   kernel_h, kernel_w,
                                   stride_h, stride_w,
                                   pad_h, pad_w,
                                   dilation_h, dilation_w,
                                   deformable_group);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

std::vector<at::Tensor>
dsp_v2_backward(const at::Tensor &input,
                const at::Tensor &offset,
                const at::Tensor &mask,
                const at::Tensor &grad_output,
                int kernel_h, int kernel_w,
                int stride_h, int stride_w,
                int pad_h, int pad_w,
                int dilation_h, int dilation_w,
                int deformable_group)
{
    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return dsp_v2_cuda_backward(input,
                                    offset,
                                    mask,
                                    grad_output,
                                    kernel_h, kernel_w,
                                    stride_h, stride_w,
                                    pad_h, pad_w,
                                    dilation_h, dilation_w,
                                    deformable_group);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}


inline void modulated_deform_conv_forward(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor ones,
    at::Tensor offset,
    at::Tensor mask,
    at::Tensor output,
    at::Tensor columns,
    int kernel_h,
    int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const int group,
    const int deformable_group,
    const bool with_bias) {
  if (input.is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
    TORCH_CHECK(weight.is_cuda(), "weight tensor is not on GPU!");
    TORCH_CHECK(bias.is_cuda(), "bias tensor is not on GPU!");
    TORCH_CHECK(offset.is_cuda(), "offset tensor is not on GPU!");
    return modulated_deform_conv_cuda_forward(
        input,
        weight,
        bias,
        ones,
        offset,
        mask,
        output,
        columns,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        group,
        deformable_group,
        with_bias);
#else
    AT_ERROR("Detectron2 is not compiled with GPU support!");
#endif
  }
  AT_ERROR("This operator is not implemented on CPU");
}

inline void modulated_deform_conv_backward(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor ones,
    at::Tensor offset,
    at::Tensor mask,
    at::Tensor columns,
    at::Tensor grad_input,
    at::Tensor grad_weight,
    at::Tensor grad_bias,
    at::Tensor grad_offset,
    at::Tensor grad_mask,
    at::Tensor grad_output,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int group,
    int deformable_group,
    const bool with_bias) {
  if (grad_output.is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
    TORCH_CHECK(input.is_cuda(), "input tensor is not on GPU!");
    TORCH_CHECK(weight.is_cuda(), "weight tensor is not on GPU!");
    TORCH_CHECK(bias.is_cuda(), "bias tensor is not on GPU!");
    TORCH_CHECK(offset.is_cuda(), "offset tensor is not on GPU!");
    return modulated_deform_conv_cuda_backward(
        input,
        weight,
        bias,
        ones,
        offset,
        mask,
        columns,
        grad_input,
        grad_weight,
        grad_bias,
        grad_offset,
        grad_mask,
        grad_output,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        group,
        deformable_group,
        with_bias);
#else
    AT_ERROR("Detectron2 is not compiled with GPU support!");
#endif
  }
  AT_ERROR("This operator is not implemented on CPU");
}