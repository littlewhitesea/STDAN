
#include "dcn_v2.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dsp_v2_forward", &dsp_v2_forward, "dsp_v2_forward");
  m.def("dsp_v2_backward", &dsp_v2_backward, "dsp_v2_backward");
  m.def("modulated_deform_conv_forward", &modulated_deform_conv_forward, "modulated_deform_conv_forward");
  m.def("modulated_deform_conv_backward", &modulated_deform_conv_backward, "modulated_deform_conv_backward");
}
