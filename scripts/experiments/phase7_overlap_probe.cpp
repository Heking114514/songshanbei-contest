// Standalone NODE-LOCAL TP probe. Never allocates a cluster resource or writes AFS.
// Usage: probe TOKENS INPUT_WIDTH OUTPUT_WIDTH RANKS REPEATS
// Link to the isolated experiment install, not the production runtime.
// All variants use software Graph replay to compare stream scheduling fairly.
#include <infinicore/context/context.hpp>
#include <infinicore/ops/linear_allreduce.hpp>
#include <infiniccl.h>
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace ic = infinicore;
static void check(infiniStatus_t status) {
    if (status != INFINI_STATUS_SUCCESS) {
        throw std::runtime_error("Infini status " + std::to_string(status));
    }
}
class Barrier {
    size_t size_, count_ = 0, generation_ = 0;
    std::mutex mutex_;
    std::condition_variable condition_;
public:
    explicit Barrier(size_t size) : size_(size) {}
    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        const auto generation = generation_;
        if (++count_ == size_) {
            count_ = 0;
            ++generation_;
            condition_.notify_all();
        } else {
            condition_.wait(lock, [&]() { return generation_ != generation; });
        }
    }
};
static uint16_t bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fff + ((bits >> 16) & 1);
    return static_cast<uint16_t>(bits >> 16);
}
static float fp32(uint16_t value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}
static size_t number(const char *text, size_t maximum) {
    if (!text || !*text) {
        throw std::invalid_argument("Missing numeric argument");
    }
    size_t value = 0;
    for (const char *p = text; *p; ++p) {
        if (*p < '0' || *p > '9' || static_cast<size_t>(*p-'0') > maximum
            || value > (maximum-static_cast<size_t>(*p-'0'))/10) {
            throw std::invalid_argument("Invalid numeric argument");
        }
        value = value*10 + static_cast<size_t>(*p-'0');
    }
    if (!value) {
        throw std::invalid_argument("Zero argument");
    }
    return value;
}
int main(int argc, char **argv) {
    try {
        if (argc != 6) {
            throw std::invalid_argument("usage: probe TOKENS INPUT_WIDTH OUTPUT_WIDTH RANKS REPEATS");
        }
        const size_t tokens = number(argv[1], 65536);
        const size_t in = number(argv[2], 32768), out = number(argv[3], 32768);
        const size_t ranks = number(argv[4], 8), repeats = number(argv[5], 100);
        if (tokens < 4096 || ranks < 2 || repeats < 3) {
            throw std::invalid_argument("Need tokens>=4096 ranks>=2 repeats>=3");
        }
        // One configuration for every rank. Lt combination is screened later.
        setenv("H3_EXPERIMENT_GEMM_LT", "0", 1);
        setenv("INFINICORE_DISABLE_DEVICE_GRAPH_SEGMENTS", "1", 1);
        check(infinirtInit());
        int devices = 0;
        check(infinirtGetDeviceCount(INFINI_DEVICE_HYGON, &devices));
        if (devices < static_cast<int>(ranks)) {
            throw std::runtime_error("Insufficient visible HYGON GPUs");
        }
        std::vector<int> ids(ranks);
        std::vector<infinicclComm_t> comms(ranks, nullptr);
        for (size_t rank = 0; rank < ranks; ++rank) {
            ids[rank] = static_cast<int>(rank);
        }
        check(infinicclCommInitAll(INFINI_DEVICE_HYGON, comms.data(),
                                  static_cast<int>(ranks), ids.data()));
        Barrier barrier(ranks);
        std::mutex log;
        std::vector<std::thread> workers;
        for (size_t rank = 0; rank < ranks; ++rank) {
            workers.emplace_back([&, rank]() {
                try {
                    const ic::Device device(ic::Device::Type::HYGON, rank);
                    ic::context::setDevice(device);
                    std::vector<uint16_t> host_input(tokens*in), host_weight(out*in);
                    for (size_t i=0; i<host_input.size(); ++i) {
                        host_input[i] = bf16(static_cast<float>(static_cast<int>((i*17+rank*3)%127)-63)/128);
                    }
                    for (size_t i=0; i<host_weight.size(); ++i) {
                        host_weight[i] = bf16(static_cast<float>(static_cast<int>((i*31+rank)%113)-56)/512);
                    }
                    auto input = ic::Tensor::from_blob(host_input.data(), {tokens,in},
                        ic::DataType::BF16, ic::Device::cpu())->to(device);
                    auto weight = ic::Tensor::from_blob(host_weight.data(), {out,in},
                        ic::DataType::BF16, ic::Device::cpu())->to(device);
                    ic::context::syncDevice();
                    std::vector<uint16_t> baseline;
                    for (const size_t chunk : {size_t(0),size_t(4096),size_t(8192),size_t(2048),size_t(1024)}) {
                        if (chunk && tokens <= chunk) {
                            continue; // The native overlap eligibility would keep old BLAS.
                        }
                        barrier.wait();
                        if (rank==0) {
                            setenv("H3_EXPERIMENT_TP_OVERLAP_ROWS", std::to_string(chunk).c_str(), 1);
                        }
                        barrier.wait();
                        ic::context::startGraphRecording();
                        auto output = ic::op::linear_allreduce(input, weight, std::nullopt, comms[rank]);
                        auto graph = ic::context::stopGraphRecording();
                        // Recording only stores operators; execute before reading the reference.
                        barrier.wait();
                        graph->run();
                        ic::context::syncDevice();
                        std::vector<uint16_t> first(tokens*out), last(tokens*out);
                        ic::context::memcpyD2H(first.data(), output->data(), first.size()*2);
                        const auto start = ic::context::createEvent(), stop = ic::context::createEvent();
                        std::vector<float> ms;
                        bool repeated = true;
                        for (size_t iteration=0; iteration<repeats; ++iteration) {
                            barrier.wait();
                            ic::context::recordEvent(start, ic::context::getStream());
                            graph->run();
                            ic::context::recordEvent(stop, ic::context::getStream());
                            ic::context::synchronizeEvent(stop);
                            ms.push_back(ic::context::elapsedTime(start, stop));
                            ic::context::memcpyD2H(last.data(), output->data(), last.size()*2);
                            repeated = repeated && first==last;
                        }
                        ic::context::destroyEvent(start);
                        ic::context::destroyEvent(stop);
                        if (chunk==0) {
                            baseline=first;
                        }
                        size_t mismatches=0;
                        double delta=0,denominator=0,max_abs=0;
                        for (size_t i=0; i<first.size(); ++i) {
                            if (first[i]!=baseline[i]) { ++mismatches; }
                            const double actual=fp32(first[i]), expected=fp32(baseline[i]);
                            if (!std::isfinite(actual) || !std::isfinite(expected)) {
                                throw std::runtime_error("Nonfinite output");
                            }
                            const double difference=actual-expected;
                            delta+=difference*difference;denominator+=expected*expected;
                            max_abs=std::max(max_abs,std::abs(difference));
                        }
                        std::sort(ms.begin(),ms.end());
                        const double median = ms.size()%2 ? ms[ms.size()/2]
                            : 0.5*(ms[ms.size()/2-1]+ms[ms.size()/2]);
                        {
                            std::lock_guard<std::mutex> lock(log);
                            std::cout << std::setprecision(10)
                                << "PHASE7_TP_RESULT {\"rank\":" << rank << ",\"chunk\":" << chunk
                                << ",\"tokens\":" << tokens << ",\"in\":" << in << ",\"out\":" << out
                                << ",\"median_ms\":" << median << ",\"bit_mismatches\":" << mismatches
                                << ",\"relative_l2\":" << std::sqrt(delta/std::max(denominator,1e-30))
                                << ",\"max_abs\":" << max_abs
                                << ",\"repeated_bitwise\":" << (repeated?"true":"false")
                                << ",\"software_graph\":true,\"lt_enabled\":false}" << std::endl;
                        }
                        if (!repeated) {
                            throw std::runtime_error("Nonreproducible repeated output");
                        }
                        ic::context::syncDevice();
                        barrier.wait();
                    }
                } catch (const std::exception &error) {
                    std::fprintf(stderr, "PHASE7_TP_FATAL rank=%zu %s\n", rank, error.what());
                    std::fflush(stderr);
                    std::_Exit(2); // Terminate all rank threads after a failed collective.
                }
            });
        }
        for (auto &worker:workers) {
            worker.join();
        }
        for (auto comm:comms) {
            check(infinicclCommDestroy(comm));
        }
        std::cout << "PHASE7_TP_COMPLETE model_acceptance=false" << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "PHASE7_TP_FATAL %s\n", error.what());
        return 2;
    }
}
