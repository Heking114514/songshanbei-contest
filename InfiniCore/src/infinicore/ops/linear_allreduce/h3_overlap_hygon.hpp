#pragma once
#include "h3_overlap_policy.hpp"
#include <infiniop/ops/gemm.h>
#include <hip/hip_runtime_api.h>
#include <algorithm>
#include <array>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>

namespace infinicore::op::h3_overlap {
inline void hip_check(hipError_t status) {
    if (status != hipSuccess) {
        throw std::runtime_error("H3 TP overlap HIP error: " + std::to_string(static_cast<int>(status)));
    }
}
struct Resources {
    int device;
    hipStream_t communication = nullptr;
    std::array<hipEvent_t, 2> ready{};
    hipEvent_t drained = nullptr;
    Tensor workspace;
    size_t workspace_bytes = 0;

    explicit Resources(int id) : device(id) {
        try {
            hip_check(hipStreamCreateWithFlags(&communication, hipStreamNonBlocking));
            for (auto &event : ready) {
                hip_check(hipEventCreateWithFlags(&event, hipEventDisableTiming));
            }
            hip_check(hipEventCreateWithFlags(&drained, hipEventDisableTiming));
        } catch (...) {
            release();
            throw;
        }
    }
    Resources(const Resources &) = delete;
    Resources &operator=(const Resources &) = delete;
    void reserve(size_t bytes) {
        if (bytes > workspace_bytes || !workspace) {
            workspace = Tensor::empty({std::max(bytes, size_t(1))}, DataType::U8, context::getDevice());
            workspace_bytes = bytes;
        }
    }
    void release() noexcept {
        int previous = -1;
        hipGetDevice(&previous);
        hipSetDevice(device);
        if (communication) {
            hipStreamSynchronize(communication);
        }
        workspace = Tensor{};
        for (auto &event : ready) {
            if (event) {
                hipEventDestroy(event);
                event = nullptr;
            }
        }
        if (drained) {
            hipEventDestroy(drained);
            drained = nullptr;
        }
        if (communication) {
            hipStreamDestroy(communication);
            communication = nullptr;
        }
        if (previous >= 0) {
            hipSetDevice(previous);
        }
    }
    ~Resources() {
        release();
    }
};
inline std::shared_ptr<Resources> resources(int device) {
    int current = -1;
    hip_check(hipGetDevice(&current));
    if (current != device) {
        throw std::invalid_argument("H3 overlap resources requested on a different device");
    }
    // One RankWorker thread owns its stream and workspace; no global scratch.
    thread_local std::map<int, std::shared_ptr<Resources>> cache;
    auto result = cache[device];
    if (!result) {
        result = std::make_shared<Resources>(device);
        cache[device] = result;
    }
    return result;
}
struct Gemm {
    infiniopGemmDescriptor_t descriptor = nullptr;
    size_t workspace_bytes = 0;
    Gemm(const Tensor &out, const Tensor &input, const Tensor &weight) {
        try {
            INFINICORE_CHECK_ERROR(infiniopCreateGemmDescriptor(
                context::getInfiniopHandle(context::getDevice()), &descriptor,
                out->desc(), input->desc(), weight->desc()));
            INFINICORE_CHECK_ERROR(infiniopGetGemmWorkspaceSize(descriptor, &workspace_bytes));
        } catch (...) {
            if (descriptor) {
                infiniopDestroyGemmDescriptor(descriptor);
            }
            throw;
        }
    }
    Gemm(const Gemm &) = delete;
    Gemm &operator=(const Gemm &) = delete;
    ~Gemm() {
        if (descriptor) {
            infiniopDestroyGemmDescriptor(descriptor);
        }
    }
};

class Overlap final : public graph::GraphOperator {
    graph::GraphTensor out_, input_, weight_;
    infinicclComm_t communicator_;
    Partition partition_;
    std::shared_ptr<Resources> resources_;
    std::map<size_t, std::shared_ptr<Gemm>> descriptors_;
    Device device_;
public:
    Overlap(const Tensor &out, const Tensor &input, const Tensor &weight,
            infinicclComm_t communicator, size_t rows)
        : out_(out), input_(input), weight_(weight), communicator_(communicator),
          partition_(input->size(0), out->size(1), input->size(1), rows),
          resources_(resources(input->device().getIndex())), device_(input->device()) {
        // At most two unique GEMM shapes: full chunk and final tail.
        for (size_t first = 0; first < partition_.rows;) {
            const size_t count = std::min(partition_.chunk, partition_.rows - first);
            if (descriptors_.find(count) == descriptors_.end()) {
                auto c = out->narrow({{0, first, count}});
                auto a = input->narrow({{0, first, count}});
                auto gemm = std::make_shared<Gemm>(c, a, weight);
                resources_->reserve(gemm->workspace_bytes);
                descriptors_.emplace(count, std::move(gemm));
            }
            first += count;
        }
        std::cerr << "H3_TP_OVERLAP_PLAN rows=" << partition_.rows
                  << " chunk=" << partition_.chunk << " width=" << partition_.width
                  << " rank_device=" << device_.getIndex()
                  << " shared_rank_workspace=" << resources_->workspace_bytes
                  << " device_graph_capture=false" << std::endl;
    }
    bool is_device_graph_capture_safe() const override {
        // Cross-stream collectives/events are explicitly submitted each replay.
        return false;
    }
    void run() const override {
        if (context::getDevice() != device_) {
            throw std::runtime_error("H3 overlap planned device changed");
        }
        const auto compute = static_cast<hipStream_t>(context::getStream());
        const auto comm = resources_->communication;
        Tensor mutable_output = out_;
        auto *out = reinterpret_cast<unsigned char *>(mutable_output->data());
        const auto *input = reinterpret_cast<const unsigned char *>(input_->data());
        try {
            partition_.submit(
                [&](size_t first, size_t count) {
                    const auto &gemm = descriptors_.at(count);
                    INFINICORE_CHECK_ERROR(infiniopGemm(
                        gemm->descriptor, resources_->workspace->data(),
                        resources_->workspace_bytes,
                        out + first * partition_.width * 2,
                        input + first * partition_.input_width * 2,
                        weight_->data(), 1.0f, 0.0f, compute));
                },
                [&](size_t slot) {
                    hip_check(hipEventRecord(resources_->ready[slot], compute));
                },
                [&](size_t first, size_t count, size_t slot) {
                    hip_check(hipStreamWaitEvent(comm, resources_->ready[slot], 0));
                    void *slice = out + first * partition_.width * 2;
                    INFINICORE_CHECK_ERROR(infinicclAllReduce(
                        slice, slice, count * partition_.width,
                        INFINI_DTYPE_BF16, INFINICCL_SUM, communicator_, comm));
                },
                [&]() {
                    hip_check(hipEventRecord(resources_->drained, comm));
                    hip_check(hipStreamWaitEvent(compute, resources_->drained, 0));
                });
        } catch (...) {
            // Prevent allocator reuse while our asynchronous writes remain live.
            // A failed rank/collective still requires the caller to abort the job.
            hipStreamSynchronize(compute);
            hipStreamSynchronize(comm);
            throw;
        }
    }
};

inline Tensor execute(Tensor out, Tensor input, Tensor weight,
                      infinicclComm_t communicator, size_t rows) {
    auto op = std::make_shared<Overlap>(out, input, weight, communicator, rows);
    if (context::isGraphRecording()) {
        context::addGraphOperator(op);
    } else {
        op->run();
    }
    return out;
}
} // namespace infinicore::op::h3_overlap
