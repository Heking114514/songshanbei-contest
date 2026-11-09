#pragma once
// Diagnostic-only host copies of AUDIO rows. No arithmetic or production-buffer writes.
#include "../../global_state/parallel_state.hpp"
#include <infinicore/context/context.hpp>
#include <infinicore/ops/embedding.hpp>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace infinilm::models::minimax_h3 {
class AudioRowTrace {
    std::filesystem::path directory_;
    size_t used_ = 0;
    size_t limit_ = 512 * 1024 * 1024;
    size_t sequence_ = 0;

public:
    AudioRowTrace() {
        const char *directory = std::getenv("H3_PHASE8_NATIVE_TRACE_DIR");
        if (!directory || !*directory
            || infinilm::global_state::get_tensor_model_parallel_rank() != 0) {
            return;
        }
        if (infinicore::context::isGraphRecording()) {
            throw std::runtime_error("Phase8 audio diagnostics require eager model orchestration");
        }
        if (!std::filesystem::path(directory).is_absolute()) {
            throw std::runtime_error("Phase8 trace directory must be an absolute path");
        }
        directory_ = std::filesystem::weakly_canonical(directory);
        if (!directory_.is_absolute() || !std::filesystem::is_directory(directory_)) {
            throw std::runtime_error("Phase8 trace directory must already exist");
        }
        if (const char *afs = std::getenv("IFAB_AFS_HOME")) {
            const auto base = std::filesystem::weakly_canonical(afs).string();
            const auto path = directory_.string();
            if (path == base || path.rfind(base + "/", 0) == 0) {
                throw std::runtime_error("Phase8 tensor traces cannot be written to AFS");
            }
        }
    }

    bool enabled() const { return !directory_.empty(); }

    void write(const std::string &name, const infinicore::Tensor &value) {
        if (!enabled()) { return; }
        const bool bf16 = value->dtype() == infinicore::DataType::BF16;
        if (!bf16 && value->dtype() != infinicore::DataType::F32) {
            throw std::runtime_error("Phase8 traces support BF16/F32 only");
        }
        const size_t element_size = bf16 ? 2 : 4;
        if (value->numel() > (limit_ - used_) / element_size) {
            throw std::runtime_error("Phase8 native trace exceeds 512MiB per selected DiT call");
        }
        const size_t bytes = value->numel() * element_size;
        const auto stem = std::to_string(sequence_++) + "." + name;
        const auto binary = directory_ / (stem + ".bin");
        const auto metadata = directory_ / (stem + ".json");
        if (std::filesystem::exists(binary) || std::filesystem::exists(metadata)) {
            throw std::runtime_error("Refusing to overwrite an existing native tensor trace");
        }
        infinicore::Tensor host;
        {
            struct RestoreDevice {
                infinicore::Device device;
                ~RestoreDevice() { infinicore::context::setDevice(device); }
            } restore{infinicore::context::getDevice()};
            infinicore::context::syncDevice();
            host = value->contiguous()->to(infinicore::Device::cpu());
            infinicore::context::syncDevice();
        }
        std::ofstream out(binary.string(), std::ios::binary);
        out.write(reinterpret_cast<const char *>(host->data()), bytes);
        out.close();
        if (!out) { throw std::runtime_error("Failed to write native audio trace"); }
        std::ofstream meta(metadata.string());
        meta << "{\"stage\":\"" << name << "\",\"dtype\":\""
             << (bf16 ? "bfloat16" : "float32") << "\",\"shape\":[";
        for (size_t i = 0; i < value->ndim(); ++i) {
            if (i) { meta << ","; }
            meta << value->size(i);
        }
        meta << "],\"bytes\":" << bytes << ",\"sequence\":" << sequence_ - 1
             << ",\"byteorder\":\"little\",\"rank\":0}\n";
        meta.close();
        if (!meta) { throw std::runtime_error("Failed to write native trace metadata"); }
        used_ += bytes;
    }

    void rows(const std::string &name, const infinicore::Tensor &packed,
              const infinicore::Tensor &audio_indices) {
        if (!enabled()) { return; }
        auto audio = infinicore::op::embedding(audio_indices, packed->squeeze(0))->unsqueeze(0);
        write(name, audio);
    }
};
} // namespace infinilm::models::minimax_h3
