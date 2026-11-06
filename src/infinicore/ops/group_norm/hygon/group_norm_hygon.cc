#if defined(ENABLE_HYGON_API) && defined(ENABLE_ATEN)
#include "infinicore/ops/group_norm.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"

#include <ATen/ops/group_norm.h>
#include <c10/hip/HIPGuard.h>

namespace infinicore::op::group_norm_impl::hygon {

void run(Tensor output,
         const Tensor &input,
         const Tensor &weight,
         const Tensor &bias,
         int64_t num_groups,
         float epsilon) {
#if INFINICORE_TORCH_VERSION_GE_2_11
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#endif
    auto input_aten = infinicore::adaptor::to_aten_tensor(input);
    auto weight_aten = infinicore::adaptor::to_aten_tensor(weight);
    auto bias_aten = infinicore::adaptor::to_aten_tensor(bias);

    const bool copy_back = !output->is_contiguous();
    Tensor output_work = copy_back ? output->contiguous() : output;
    auto output_aten = infinicore::adaptor::to_aten_tensor(output_work);
    auto result = at::group_norm(
        input_aten,
        num_groups,
        std::optional<at::Tensor>(weight_aten),
        std::optional<at::Tensor>(bias_aten),
        static_cast<double>(epsilon),
        true);
    output_aten.copy_(result);
    if (copy_back) {
        output->copy_from(output_work);
    }
}

static bool registered = []() {
    GroupNorm::dispatcher().registerDevice(Device::Type::HYGON, &run);
    return true;
}();

} // namespace infinicore::op::group_norm_impl::hygon
#endif // ENABLE_HYGON_API && ENABLE_ATEN
