#ifndef H3_LT_POLICY_H
#define H3_LT_POLICY_H

// HYGON experiment only. No GPU calls, timing or operand mutation here.
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>

namespace op::gemm::nvidia::h3_lt {

constexpr unsigned kMaxCandidates = 16;
constexpr size_t kMaxTableBytes = 256 * 1024;
constexpr size_t kMaxTableRules = 4096;

inline bool unsignedValue(const char *text, size_t limit, size_t &out) {
    if (!text || !*text) {
        return false;
    }
    size_t value = 0;
    for (const char *p = text; *p; ++p) {
        if (*p < '0' || *p > '9') {
            return false;
        }
        const size_t digit = static_cast<size_t>(*p - '0');
        if (digit > limit || value > (limit - digit) / 10) {
            return false;
        }
        value = value * 10 + digit;
    }
    out = value;
    return true;
}

inline bool token(const std::string &value, size_t limit) {
    if (value.empty() || value.size() > limit) {
        return false;
    }
    for (unsigned char ch : value) {
        if (!((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z')
              || (ch >= '0' && ch <= '9') || ch == '_' || ch == '-'
              || ch == '.' || ch == ':')) {
            return false;
        }
    }
    return true;
}

inline bool envNumber(const char *name, size_t limit, size_t &out, std::string &error) {
    const char *value = std::getenv(name);
    if (!value) {
        return true;
    }
    if (unsignedValue(value, limit, out)) {
        return true;
    }
    error = name;
    return false;
}

struct Config {
    bool enabled = false;
    bool trace = false;
    bool table_only = true;
    size_t workspace_limit = size_t(32) << 20;
    size_t index = 0;
    size_t min_tokens = 4096;
    size_t alignment = 16;
    std::string runtime_tag = "unbound";
    std::string table_path;

    static bool read(Config &out, std::string &error) {
        size_t enabled = 0, trace = 0;
        if (!envNumber("H3_EXPERIMENT_GEMM_LT", 1, enabled, error)
            || !envNumber("H3_GEMM_LT_TRACE", 1, trace, error)) {
            return false;
        }
        out.enabled = enabled != 0;
        out.trace = trace != 0;
        // Disabled means the old HYGON BLAS path, even if tuning env is stale.
        if (!out.enabled) {
            return true;
        }
        size_t mb = 32, table_only = 1;
        if (!envNumber("H3_GEMM_LT_WORKSPACE_MB", 256, mb, error)
            || !envNumber("H3_GEMM_LT_INDEX", kMaxCandidates - 1, out.index, error)
            || !envNumber("H3_GEMM_LT_MIN_TOKENS", 2147483647, out.min_tokens, error)
            || !envNumber("H3_GEMM_LT_ALIGNMENT", 256, out.alignment, error)
            || !envNumber("H3_GEMM_LT_TABLE_ONLY", 1, table_only, error)) {
            return false;
        }
        if (out.min_tokens == 0 || out.alignment < 2
            || (out.alignment & (out.alignment - 1)) != 0) {
            error = "min_tokens_or_alignment";
            return false;
        }
        out.workspace_limit = mb << 20;
        out.table_only = table_only != 0;
        if (const char *tag = std::getenv("H3_GEMM_LT_RUNTIME_TAG")) {
            out.runtime_tag = tag;
            if (!token(out.runtime_tag, 128)) {
                error = "H3_GEMM_LT_RUNTIME_TAG";
                return false;
            }
        }
        if (const char *path = std::getenv("H3_GEMM_LT_TABLE")) {
            out.table_path = path;
            if (out.table_path.empty() || out.table_path.size() > 4096
                || out.runtime_tag == "unbound") {
                error = "table_requires_runtime_fingerprint";
                return false;
            }
        }
        return true;
    }

    bool eligible(size_t m, size_t n, size_t k, size_t batch) const {
        // COL_MAJOR MatmulInfo: n is the token count for row-major H3 linear.
        return enabled && batch == 1 && n >= min_tokens && m >= 128 && k >= 128
            && m <= 2147483647 && n <= 2147483647 && k <= 2147483647;
    }
};

inline bool parseTable(const std::string &text, std::map<std::string, int> &out,
                       std::string &error) {
    if (text.size() > kMaxTableBytes) {
        error = "table_too_large";
        return false;
    }
    std::map<std::string, int> parsed;
    std::istringstream input(text);
    std::string line;
    size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        line = line.substr(0, line.find('#'));
        std::istringstream fields(line);
        std::string key, eq, value, extra;
        if (!(fields >> key)) {
            continue;
        }
        size_t index = 0;
        if (!(fields >> eq >> value) || fields >> extra || eq != "="
            || !token(key, 1024)
            || !unsignedValue(value.c_str(), kMaxCandidates - 1, index)
            || !parsed.emplace(key, static_cast<int>(index)).second
            || parsed.size() > kMaxTableRules) {
            error = "table_line_" + std::to_string(line_number);
            return false;
        }
    }
    out.swap(parsed); // Never expose a partial table.
    return true;
}

inline bool tableChoice(const Config &config, const std::string &key,
                        int &choice, bool &matched, std::string &error) {
    choice = static_cast<int>(config.index);
    matched = false;
    if (config.table_path.empty()) {
        return true;
    }
    // C++11 thread-safe initialization; the snapshot is immutable thereafter.
    struct Snapshot {
        std::string path, failure;
        std::map<std::string, int> entries;
    };
    static const Snapshot snapshot = [&config]() {
        Snapshot result;
        result.path = config.table_path;
        std::ifstream input(result.path, std::ios::binary);
        if (!input) {
            result.failure = "table_open_failed";
        } else {
            std::string text;
            char buffer[4096];
            while (input && text.size() <= kMaxTableBytes) {
                input.read(buffer, sizeof(buffer));
                text.append(buffer, static_cast<size_t>(input.gcount()));
            }
            if (input.bad()) {
                result.failure = "table_read_failed";
            } else {
                parseTable(text, result.entries, result.failure);
            }
        }
        return result;
    }();
    if (snapshot.path != config.table_path || !snapshot.failure.empty()) {
        error = snapshot.path != config.table_path ? "table_path_changed" : snapshot.failure;
        return false;
    }
    const auto &table = snapshot.entries;
    const auto found = table.find(key);
    if (found != table.end()) {
        choice = found->second;
        matched = true;
    }
    return true;
}

} // namespace op::gemm::nvidia::h3_lt
#endif
