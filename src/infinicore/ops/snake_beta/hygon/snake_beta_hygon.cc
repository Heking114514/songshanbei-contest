#if defined(ENABLE_HYGON_API) && defined(ENABLE_ATEN)
#include "infinicore/ops/snake_beta.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"

#include <ATen/ATen.h>
#include <c10/hip/HIPGuard.h>

namespace infinicore::op::snake_beta_impl::hygon {

void run(Tensor output,
         const Tensor &input,
         const Tensor &alpha_log,
         const Tensor &beta_log) {
#if INFINICORE_TORCH_VERSION_GE_2_11
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#endif
    auto input_aten = infinicore::adaptor::to_aten_tensor(input);
    auto alpha_aten = infinicore::adaptor::to_aten_tensor(alpha_log);
    auto beta_aten = infinicore::adaptor::to_aten_tensor(beta_log);
    const auto channels = input_aten.size(1);

    auto alpha = alpha_aten.reshape({1, channels, 1}).exp();
    auto beta = beta_aten.reshape({1, channels, 1}).exp();
    auto result = input_aten
                + at::sin(alpha * input_aten).pow(2)
                    / (beta + 1e-9);

    const bool copy_back = !output->is_contiguous();
    Tensor output_work = copy_back ? output->contiguous() : output;
    auto output_aten = infinicore::adaptor::to_aten_tensor(output_work);
    output_aten.copy_(result);
    if (copy_back) {
        output->copy_from(output_work);
    }
}

static bool registered = []() {
    SnakeBeta::dispatcher().registerDevice(Device::Type::HYGON, &run);
    return true;
}();

} // namespace infinicore::op::snake_beta_impl::hygon
#endif // ENABLE_HYGON_API && ENABLE_ATEN
