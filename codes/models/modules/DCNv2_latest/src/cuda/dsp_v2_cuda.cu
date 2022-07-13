#include <vector>
#include "dcn_v2_im2col_cuda.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

// THCState *state = at::globalContext().lazyInitCUDA();

// author: Charles Shang
// https://github.com/torch/cunn/blob/master/lib/THCUNN/generic/SpatialConvolutionMM.cu

// [batch gemm]
// https://github.com/pytorch/pytorch/blob/master/aten/src/THC/generic/THCTensorMathBlas.cu


at::Tensor
dsp_v2_cuda_forward(const at::Tensor &input,
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
    using scalar_t = float;
    AT_ASSERTM(input.is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(offset.is_cuda(), "offset must be a CUDA tensor");
    AT_ASSERTM(mask.is_cuda(), "mask must be a CUDA tensor");

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    auto output = at::empty({batch, channels * kernel_h * kernel_w, 1 * height * width}, input.options());


    modulated_deformable_im2col_cuda(c10::cuda::getCurrentCUDAStream(),
                                 input.data_ptr<scalar_t>(),
                                 offset.data_ptr<scalar_t>(),
                                 mask.data_ptr<scalar_t>(),
                                 batch, channels, height, width,
                                 height, width, kernel_h, kernel_w,
                                 pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                                 deformable_group,
                                 output.data_ptr<scalar_t>());

    output = output.view({batch, channels, kernel_h * kernel_w, height * width});

    return output;
}

std::vector<at::Tensor> dsp_v2_cuda_backward(const at::Tensor &input,
                                             const at::Tensor &offset,
                                             const at::Tensor &mask,
                                             const at::Tensor &grad_output,
                                             int kernel_h, int kernel_w,
                                             int stride_h, int stride_w,
                                             int pad_h, int pad_w,
                                             int dilation_h, int dilation_w,
                                             int deformable_group)
{

    THArgCheck(input.is_contiguous(), 1, "input tensor has to be contiguous");

    AT_ASSERTM(input.is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(offset.is_cuda(), "offset must be a CUDA tensor");
    AT_ASSERTM(mask.is_cuda(), "mask must be a CUDA tensor");

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    auto output = at::empty({batch, channels, kernel_h * kernel_w, height * width}, input.options());

    auto grad_input = at::zeros_like(input);
    auto grad_offset = at::zeros_like(offset);
    auto grad_mask = at::zeros_like(mask);

    using scalar_t = float;

    for (int b = 0; b < batch; b++)
    {
        auto input_n = input.select(0, b);
        auto offset_n = offset.select(0, b);
        auto mask_n = mask.select(0, b);
        auto grad_output_n = grad_output.select(0, b);
        auto grad_input_n = grad_input.select(0, b);
        auto grad_offset_n = grad_offset.select(0, b);
        auto grad_mask_n = grad_mask.select(0, b);

        // auto grad_output_n_flat = grad_output_n.view({channels*kernel_h*kernel_w, height*width});
        auto grad_output_n_flat = grad_output_n.reshape({channels*kernel_h*kernel_w, height*width});

        // gradient w.r.t. input coordinate data
        modulated_deformable_col2im_coord_cuda(c10::cuda::getCurrentCUDAStream(),
                                               grad_output_n_flat.data_ptr<scalar_t>(),
                                               input_n.data_ptr<scalar_t>(),
                                               offset_n.data_ptr<scalar_t>(),
                                               mask_n.data_ptr<scalar_t>(),
                                               1, channels, height, width,
                                               height, width, kernel_h, kernel_w,
                                               pad_h, pad_w, stride_h, stride_w,
                                               dilation_h, dilation_w, deformable_group,
                                               grad_offset_n.data_ptr<scalar_t>(),
                                               grad_mask_n.data_ptr<scalar_t>());
        // gradient w.r.t. input data
        modulated_deformable_col2im_cuda(c10::cuda::getCurrentCUDAStream(),
                                         grad_output_n_flat.data_ptr<scalar_t>(),
                                         offset_n.data_ptr<scalar_t>(),
                                         mask_n.data_ptr<scalar_t>(),
                                         1, channels, height, width,
                                         height, width, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w,
                                         dilation_h, dilation_w, deformable_group,
                                         grad_input_n.data_ptr<scalar_t>());

    }

    return {
        grad_input, grad_offset, grad_mask
    };
}


void modulated_deformable_im2col_cuda2(
    const at::Tensor data_im,
    const at::Tensor data_offset,
    const at::Tensor data_mask,
    const int batch_size,
    const int channels,
    const int height_im,
    const int width_im,
    const int height_col,
    const int width_col,
    const int kernel_h,
    const int kenerl_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int deformable_group,
    at::Tensor data_col);

void modulated_deformable_col2im_cuda2(
    const at::Tensor data_col,
    const at::Tensor data_offset,
    const at::Tensor data_mask,
    const int batch_size,
    const int channels,
    const int height_im,
    const int width_im,
    const int height_col,
    const int width_col,
    const int kernel_h,
    const int kenerl_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int deformable_group,
    at::Tensor grad_im);

void modulated_deformable_col2im_coord_cuda2(
    const at::Tensor data_col,
    const at::Tensor data_im,
    const at::Tensor data_offset,
    const at::Tensor data_mask,
    const int batch_size,
    const int channels,
    const int height_im,
    const int width_im,
    const int height_col,
    const int width_col,
    const int kernel_h,
    const int kenerl_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int deformable_group,
    at::Tensor grad_offset,
    at::Tensor grad_mask);

void shape_check(
    at::Tensor input,
    at::Tensor offset,
    at::Tensor* gradOutput,
    at::Tensor weight,
    int kH,
    int kW,
    int dH,
    int dW,
    int padH,
    int padW,
    int dilationH,
    int dilationW,
    int group,
    int deformable_group) {
  TORCH_CHECK(
      weight.ndimension() == 4,
      "4D weight tensor (nOutputPlane,nInputPlane,kH,kW) expected, "
      "but got: %s",
      weight.ndimension());

  TORCH_CHECK(weight.is_contiguous(), "weight tensor has to be contiguous");

  TORCH_CHECK(
      kW > 0 && kH > 0,
      "kernel size should be greater than zero, but got kH: %d kW: %d",
      kH,
      kW);

  TORCH_CHECK(
      (weight.size(2) == kH && weight.size(3) == kW),
      "kernel size should be consistent with weight, ",
      "but got kH: %d kW: %d weight.size(2): %d, weight.size(3): %d",
      kH,
      kW,
      weight.size(2),
      weight.size(3));

  TORCH_CHECK(
      dW > 0 && dH > 0,
      "stride should be greater than zero, but got dH: %d dW: %d",
      dH,
      dW);

  TORCH_CHECK(
      dilationW > 0 && dilationH > 0,
      "dilation should be greater than 0, but got dilationH: %d dilationW: %d",
      dilationH,
      dilationW);

  int ndim = input.ndimension();
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  TORCH_CHECK(
      ndim == 3 || ndim == 4,
      "3D or 4D input tensor expected but got: %s",
      ndim);

  long nInputPlane = weight.size(1) * group;
  long inputHeight = input.size(dimh);
  long inputWidth = input.size(dimw);
  long nOutputPlane = weight.size(0);
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;

  TORCH_CHECK(
      nInputPlane % deformable_group == 0,
      "input channels must divide deformable group size");

  if (outputWidth < 1 || outputHeight < 1)
    AT_ERROR(
        "Given input size: (%ld x %ld x %ld). "
        "Calculated output size: (%ld x %ld x %ld). Output size is too small",
        nInputPlane,
        inputHeight,
        inputWidth,
        nOutputPlane,
        outputHeight,
        outputWidth);

  TORCH_CHECK(
      input.size(1) == nInputPlane,
      "invalid number of input planes, expected: %d, but got: %d",
      nInputPlane,
      input.size(1));

  TORCH_CHECK(
      (inputHeight + 2 * padH >= kH && inputWidth + 2 * padW >= kW),
      "input image is smaller than kernel");

  TORCH_CHECK(
      (offset.size(2) == outputHeight && offset.size(3) == outputWidth),
      "invalid spatial size of offset, expected height: %d width: %d, but "
      "got height: %d width: %d",
      outputHeight,
      outputWidth,
      offset.size(2),
      offset.size(3));

  TORCH_CHECK(
      (offset.size(1) == deformable_group * 2 * kH * kW),
      "invalid number of channels of offset");

  if (gradOutput != NULL) {
    TORCH_CHECK(
        gradOutput->size(dimf) == nOutputPlane,
        "invalid number of gradOutput planes, expected: %d, but got: %d",
        nOutputPlane,
        gradOutput->size(dimf));

    TORCH_CHECK(
        (gradOutput->size(dimh) == outputHeight &&
         gradOutput->size(dimw) == outputWidth),
        "invalid size of gradOutput, expected height: %d width: %d , but "
        "got height: %d width: %d",
        outputHeight,
        outputWidth,
        gradOutput->size(dimh),
        gradOutput->size(dimw));
  }
}


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
    const bool with_bias) {
  shape_check(
      input,
      offset,
      NULL,
      weight,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      dilation_h,
      dilation_w,
      group,
      deformable_group);

  TORCH_CHECK(input.is_contiguous(), "input tensor has to be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight tensor has to be contiguous");

  const int batch = input.size(0);
  const int channels = input.size(1);
  const int height = input.size(2);
  const int width = input.size(3);

  const int channels_out = weight.size(0);
  const int channels_kernel = weight.size(1);
  const int kernel_h_ = weight.size(2);
  const int kernel_w_ = weight.size(3);

  if (kernel_h_ != kernel_h || kernel_w_ != kernel_w)
    AT_ERROR(
        "Input shape and kernel shape wont match: (%d x %d vs %d x %d).",
        kernel_h_,
        kernel_w,
        kernel_h_,
        kernel_w_);
  if (channels != channels_kernel * group)
    AT_ERROR(
        "Input shape and kernel channels wont match: (%d vs %d).",
        channels,
        channels_kernel * group);

  const int height_out =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int width_out =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  // mask shape check
  TORCH_CHECK(
      (mask.size(2) == height_out && mask.size(3) == width_out),
      "invalid spatial size of mask, expected height: %d width: %d, but "
      "got height: %d width: %d",
      height_out,
      width_out,
      mask.size(2),
      mask.size(3));

  TORCH_CHECK(
      (mask.size(1) == deformable_group * kernel_h * kernel_w),
      "invalid number of channels of mask");

  if (ones.ndimension() != 2 ||
      ones.size(0) * ones.size(1) < height_out * width_out) {
    // Resize plane and fill with ones...
    ones = at::ones({height_out, width_out}, input.options());
  }

  // resize output
  output = output.view({batch, channels_out, height_out, width_out}).zero_();
  // resize temporary columns
  columns = at::zeros(
      {channels * kernel_h * kernel_w, 1 * height_out * width_out},
      input.options());

  output = output.view(
      {output.size(0),
       group,
       output.size(1) / group,
       output.size(2),
       output.size(3)});

  for (int b = 0; b < batch; b++) {
    modulated_deformable_im2col_cuda2(
        input[b],
        offset[b],
        mask[b],
        1,
        channels,
        height,
        width,
        height_out,
        width_out,
        kernel_h,
        kernel_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        deformable_group,
        columns);

    // divide into group
    weight = weight.view(
        {group,
         weight.size(0) / group,
         weight.size(1),
         weight.size(2),
         weight.size(3)});
    columns = columns.view({group, columns.size(0) / group, columns.size(1)});

    for (int g = 0; g < group; g++) {
      output[b][g] = output[b][g]
                         .flatten(1)
                         .addmm_(weight[g].flatten(1), columns[g])
                         .view_as(output[b][g]);
    }

    weight = weight.view(
        {weight.size(0) * weight.size(1),
         weight.size(2),
         weight.size(3),
         weight.size(4)});
    columns =
        columns.view({columns.size(0) * columns.size(1), columns.size(2)});
  }

  output = output.view(
      {output.size(0),
       output.size(1) * output.size(2),
       output.size(3),
       output.size(4)});

  if (with_bias) {
    output += bias.view({1, bias.size(0), 1, 1});
  }
}

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
    const bool with_bias) {
  shape_check(
      input,
      offset,
      &grad_output,
      weight,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      dilation_h,
      dilation_w,
      group,
      deformable_group);

  TORCH_CHECK(input.is_contiguous(), "input tensor has to be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight tensor has to be contiguous");

  const int batch = input.size(0);
  const int channels = input.size(1);
  const int height = input.size(2);
  const int width = input.size(3);

  const int channels_kernel = weight.size(1);
  const int kernel_h_ = weight.size(2);
  const int kernel_w_ = weight.size(3);
  if (kernel_h_ != kernel_h || kernel_w_ != kernel_w)
    AT_ERROR(
        "Input shape and kernel shape wont match: (%d x %d vs %d x %d).",
        kernel_h_,
        kernel_w,
        kernel_h_,
        kernel_w_);
  if (channels != channels_kernel * group)
    AT_ERROR(
        "Input shape and kernel channels wont match: (%d vs %d).",
        channels,
        channels_kernel * group);

  const int height_out =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int width_out =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  // mask shape check
  TORCH_CHECK(
      (mask.size(2) == height_out && mask.size(3) == width_out),
      "invalid spatial size of mask, expected height: %d width: %d, but "
      "got height: %d width: %d",
      height_out,
      width_out,
      mask.size(2),
      mask.size(3));

  TORCH_CHECK(
      (mask.size(1) == deformable_group * kernel_h * kernel_w),
      "invalid number of channels of mask");

  if (ones.ndimension() != 2 ||
      ones.size(0) * ones.size(1) < height_out * width_out) {
    // Resize plane and fill with ones...
    ones = at::ones({height_out, width_out}, input.options());
  }

  grad_input = grad_input.view({batch, channels, height, width});
  columns = at::zeros(
      {channels * kernel_h * kernel_w, height_out * width_out},
      input.options());

  grad_output = grad_output.view(
      {grad_output.size(0),
       group,
       grad_output.size(1) / group,
       grad_output.size(2),
       grad_output.size(3)});

  for (int b = 0; b < batch; b++) {
    // divide int group
    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
    weight = weight.view(
        {group,
         weight.size(0) / group,
         weight.size(1),
         weight.size(2),
         weight.size(3)});

    for (int g = 0; g < group; g++) {
      columns[g].addmm_(
          weight[g].flatten(1).transpose(0, 1),
          grad_output[b][g].flatten(1),
          0.0f,
          1.0f);
    }

    columns =
        columns.view({columns.size(0) * columns.size(1), columns.size(2)});
    weight = weight.view(
        {weight.size(0) * weight.size(1),
         weight.size(2),
         weight.size(3),
         weight.size(4)});

    // gradient w.r.t. input coordinate data
    modulated_deformable_col2im_coord_cuda2(
        columns,
        input[b],
        offset[b],
        mask[b],
        1,
        channels,
        height,
        width,
        height_out,
        width_out,
        kernel_h,
        kernel_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        deformable_group,
        grad_offset[b],
        grad_mask[b]);
    // gradient w.r.t. input data
    modulated_deformable_col2im_cuda2(
        columns,
        offset[b],
        mask[b],
        1,
        channels,
        height,
        width,
        height_out,
        width_out,
        kernel_h,
        kernel_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        deformable_group,
        grad_input[b]);

    // gradient w.r.t. weight, dWeight should accumulate across the batch and
    // group
    modulated_deformable_im2col_cuda2(
        input[b],
        offset[b],
        mask[b],
        1,
        channels,
        height,
        width,
        height_out,
        width_out,
        kernel_h,
        kernel_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        deformable_group,
        columns);

    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
    grad_weight = grad_weight.view(
        {group,
         grad_weight.size(0) / group,
         grad_weight.size(1),
         grad_weight.size(2),
         grad_weight.size(3)});
    if (with_bias)
      grad_bias = grad_bias.view({group, grad_bias.size(0) / group});

    for (int g = 0; g < group; g++) {
      grad_weight[g] =
          grad_weight[g]
              .flatten(1)
              .addmm_(grad_output[b][g].flatten(1), columns[g].transpose(0, 1))
              .view_as(grad_weight[g]);
      if (with_bias) {
        grad_bias[g] =
            grad_bias[g]
                .view({-1, 1})
                .addmm_(grad_output[b][g].flatten(1), ones.view({-1, 1}))
                .view(-1);
      }
    }

    columns =
        columns.view({columns.size(0) * columns.size(1), columns.size(2)});
    grad_weight = grad_weight.view(
        {grad_weight.size(0) * grad_weight.size(1),
         grad_weight.size(2),
         grad_weight.size(3),
         grad_weight.size(4)});
    if (with_bias)
      grad_bias = grad_bias.view({grad_bias.size(0) * grad_bias.size(1)});
  }
  grad_output = grad_output.view(
      {grad_output.size(0) * grad_output.size(1),
       grad_output.size(2),
       grad_output.size(3),
       grad_output.size(4)});
}
