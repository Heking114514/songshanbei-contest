#if defined(ENABLE_HYGON_API) && defined(ENABLE_ATEN)
#include "infinicore/ops/conv_transpose1d.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"

#include <ATen/ops/conv_transpose1d.h>
#include <c10/hip/HIPGuard.h>

namespace infinicore::op::conv_transpose1d_impl::hygon {

void run(Tensor output,
         const Tensor &input,
         const Tensor &weight,
         const Tensor &bias,
         int64_t stride,
         int64_t padding,
         int64_t output_padding,
         int64_t groups,
         int64_t dilation) {
#if INFINICORE_TORCH_VERSION_GE_2_11
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#endif
    auto input_aten = infinicore::adaptor::to_aten_tensor(input);
    auto weight_aten = infinicore::adaptor::to_aten_tensor(weight);
    std::optional<at::Tensor> bias_aten;
    if (bias->numel() != 0) {
        bias_aten = infinicore::adaptor::to_aten_tensor(bias);
    }

    auto result = at::conv_transpose1d(
        input_aten,
        weight_aten,
        bias_aten,
        {stride},
        {padding},
        {output_padding},
        groups,
        {dilation});

    const bool copy_back = !output->is_contiguous();
    Tensor output_work = copy_back ? output->contiguous() : output;
    auto output_aten = infinicore::adaptor::to_aten_tensor(output_work);
    output_aten.copy_(result);
    if (copy_back) {
        output->copy_from(output_work);
    }
}

static bool registered = []() {
    ConvTranspose1d::dispatcher().registerDevice(Device::Type::HYGON, &run);
    return true;
}();

} // namespace infinicore::op::conv_transpose1d_impl::hygon
#endif // ENABLE_HYGON_API && ENABLE_ATEN
