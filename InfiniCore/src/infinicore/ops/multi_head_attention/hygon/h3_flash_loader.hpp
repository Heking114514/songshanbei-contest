#pragma once
#include "h3_flash_abi.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace infinicore::op::mha_impl::flashattn::h3_flash {
struct Library {
    void *so = nullptr;
    const H3FlashApi *api = nullptr;
    std::string path;
    explicit Library(const char *value) : path(value ? value : "") {
        if (path.empty() || path[0] != '/') {
            throw std::invalid_argument("H3_FLASH_LIBRARY must be an absolute node-local path");
        }
        so = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!so) {
            const char *reason = dlerror();
            throw std::runtime_error(std::string("H3 flash dlopen: ") + (reason ? reason : "unknown"));
        }
        auto get = reinterpret_cast<H3FlashGetApi>(dlsym(so, "h3FlashGetApi"));
        api = get ? get() : nullptr;
        if (!api || api->abi_version != H3_FLASH_ABI_VERSION
            || api->struct_bytes != sizeof(H3FlashApi) || !api->arch
            || std::strcmp(api->arch, "gfx936") != 0 || !api->build_id
            || !api->prepare || !api->launch || !api->destroy) {
            dlclose(so);
            so = nullptr;
            throw std::runtime_error("H3 flash incompatible ABI or target");
        }
    }
    ~Library() {
        if (so) {
            dlclose(so);
        }
    }
};
inline std::shared_ptr<Library> library() {
    // Immutable after first use; never unload while any prepared Graph holds it.
    static const auto loaded = std::make_shared<Library>(std::getenv("H3_FLASH_LIBRARY"));
    const char *current = std::getenv("H3_FLASH_LIBRARY");
    if (!current || loaded->path != current) {
        throw std::invalid_argument("H3_FLASH_LIBRARY changed after first plan");
    }
    return loaded;
}

struct Plan {
    std::shared_ptr<Library> owner;
    void *state = nullptr;
    H3FlashSpec spec{};
    explicit Plan(const H3FlashSpec &value) : owner(library()), spec(value) {
        const int status = owner->api->prepare(&spec, &state);
        if (status || !state) {
            if (state) {
                owner->api->destroy(state);
                state = nullptr;
            }
            throw std::runtime_error("H3 gfx936 flash prepare failed: " + std::to_string(status));
        }
        std::cerr << "H3_FLASH_PLAN backend=triton_aot target=gfx936 build="
                  << owner->api->build_id << " q=" << spec.query_tokens
                  << " kv=" << spec.key_tokens << " heads=" << spec.heads << std::endl;
    }
    Plan(const Plan &) = delete;
    Plan &operator=(const Plan &) = delete;
    ~Plan() {
        if (state) {
            owner->api->destroy(state);
        }
    }
    void run(const H3FlashCall &call, void *stream) const {
        if (!call.q || !call.k || !call.v || !call.out || !std::isfinite(call.scale)) {
            throw std::invalid_argument("H3 flash null pointer or nonfinite scale");
        }
        const int status = owner->api->launch(state, &call, stream);
        if (status) {
            // Never silently run math after a potentially partially written output.
            throw std::runtime_error("H3 gfx936 flash launch failed: " + std::to_string(status));
        }
    }
};

inline bool enabled() {
    const char *value = std::getenv("H3_EXPERIMENT_FLASH_GFX936");
    if (!value || std::strcmp(value, "0") == 0) {
        return false;
    }
    if (std::strcmp(value, "1") != 0) {
        throw std::invalid_argument("H3_EXPERIMENT_FLASH_GFX936 must be 0 or 1");
    }
    return true;
}
} // namespace infinicore::op::mha_impl::flashattn::h3_flash
