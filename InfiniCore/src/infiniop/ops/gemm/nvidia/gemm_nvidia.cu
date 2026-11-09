#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "gemm_nvidia.cuh"
#if !defined(ENABLE_ILUVATAR_API)
#include <cublasLt.h>
#endif

#ifdef ENABLE_HYGON_API
#include "h3_lt_policy.h"
#include <cstdint>
#include <cstdio>
#endif

namespace op::gemm::nvidia {

#ifdef ENABLE_HYGON_API
// Diagnostics only; the returned string lasts until this thread creates its
// next GEMM descriptor. No descriptor cache or GPU context is modified.
static thread_local std::string h3_lt_receipt = "{}";

__INFINI_C __export const char *h3GemmLtBuildId() {
    return "h3_hygon_lt_v1";
}
__INFINI_C __export const char *h3GemmLtLastSelection() {
    return h3_lt_receipt.c_str();
}

static void h3LtReceipt(const h3_lt::Config &config, const std::string &key,
                        const char *backend, const char *reason, int index,
                        int found, size_t workspace) {
    std::ostringstream out;
    // All string fields are internally generated identifier tokens.
    out << "{\"backend\":\"" << backend << "\",\"reason\":\"" << reason
        << "\",\"key\":\"" << key << "\",\"index\":" << index
        << ",\"found\":" << found << ",\"workspace\":" << workspace << "}";
    h3_lt_receipt = out.str();
    if (config.trace) {
        std::fprintf(stderr, "H3_GEMM_LT_SELECTION %s\n", h3_lt_receipt.c_str());
    }
}
#endif

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
#if !defined(ENABLE_ILUVATAR_API)
    cublasLtHandle_t lt_handle = nullptr;
    cublasLtMatmulDesc_t lt_desc = nullptr;
    cublasLtMatrixLayout_t a_layout = nullptr;
    cublasLtMatrixLayout_t b_layout = nullptr;
    cublasLtMatrixLayout_t c_layout = nullptr;

#ifdef ENABLE_HYGON_API
    cublasLtMatmulAlgo_t lt_algo{};
    bool lt_has_algo = false;
    size_t lt_workspace = 0;
    size_t lt_alignment = 16;
    std::string lt_key;
    infiniStatus_t selectHygonLt(const MatmulInfo &, const h3_lt::Config &, int);
#endif
    void destroyLtDescriptors();
    bool createBf16LtDescriptors(const MatmulInfo &info);
#endif
};

#if !defined(ENABLE_ILUVATAR_API)
static size_t ltLayoutRows(const BlasMatrix &matrix) {
    return matrix.row_stride == 1 ? matrix.rows : matrix.cols;
}

static size_t ltLayoutCols(const BlasMatrix &matrix) {
    return matrix.row_stride == 1 ? matrix.cols : matrix.rows;
}

static bool setLtLayoutBatch(cublasLtMatrixLayout_t layout, const BlasMatrix &matrix, size_t batch) {
    int32_t batch_count = static_cast<int32_t>(batch);
    int64_t stride = static_cast<int64_t>(matrix.stride);
    return cublasLtMatrixLayoutSetAttribute(
               layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
               &batch_count, sizeof(batch_count))
            == CUBLAS_STATUS_SUCCESS
        && cublasLtMatrixLayoutSetAttribute(
               layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
               &stride, sizeof(stride))
               == CUBLAS_STATUS_SUCCESS;
}

void Descriptor::Opaque::destroyLtDescriptors() {
    if (a_layout) {
        cublasLtMatrixLayoutDestroy(a_layout);
        a_layout = nullptr;
    }
    if (b_layout) {
        cublasLtMatrixLayoutDestroy(b_layout);
        b_layout = nullptr;
    }
    if (c_layout) {
        cublasLtMatrixLayoutDestroy(c_layout);
        c_layout = nullptr;
    }
    if (lt_desc) {
        cublasLtMatmulDescDestroy(lt_desc);
        lt_desc = nullptr;
    }
    if (lt_handle) {
        cublasLtDestroy(lt_handle);
        lt_handle = nullptr;
    }
}

bool Descriptor::Opaque::createBf16LtDescriptors(const MatmulInfo &info) {
    auto op_a = info.a_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;
    auto op_b = info.b_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;

    if (cublasLtCreate(&lt_handle) != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    if (cublasLtMatmulDescCreate(&lt_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    if (cublasLtMatmulDescSetAttribute(
            lt_desc, CUBLASLT_MATMUL_DESC_TRANSA,
            &op_a, sizeof(op_a))
        != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    if (cublasLtMatmulDescSetAttribute(
            lt_desc, CUBLASLT_MATMUL_DESC_TRANSB,
            &op_b, sizeof(op_b))
        != CUBLAS_STATUS_SUCCESS) {
        return false;
    }

    if (cublasLtMatrixLayoutCreate(
            &a_layout, CUDA_R_16BF,
            ltLayoutRows(info.a_matrix), ltLayoutCols(info.a_matrix),
            info.a_matrix.ld())
        != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    if (cublasLtMatrixLayoutCreate(
            &b_layout, CUDA_R_16BF,
            ltLayoutRows(info.b_matrix), ltLayoutCols(info.b_matrix),
            info.b_matrix.ld())
        != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    if (cublasLtMatrixLayoutCreate(
            &c_layout, CUDA_R_16BF,
            ltLayoutRows(info.c_matrix), ltLayoutCols(info.c_matrix),
            info.c_matrix.ld())
        != CUBLAS_STATUS_SUCCESS) {
        return false;
    }

    return setLtLayoutBatch(a_layout, info.a_matrix, info.batch)
        && setLtLayoutBatch(b_layout, info.b_matrix, info.batch)
        && setLtLayoutBatch(c_layout, info.c_matrix, info.batch);
}
#endif

#ifdef ENABLE_HYGON_API
static std::string h3DeviceToken(const char *name) {
    std::string result;
    for (const unsigned char *p = reinterpret_cast<const unsigned char *>(name); *p; ++p) {
        const bool alnum = (*p >= 'A' && *p <= 'Z') || (*p >= 'a' && *p <= 'z')
            || (*p >= '0' && *p <= '9');
        result += alnum ? static_cast<char>(*p) : '_';
    }
    return result.empty() ? "unknown" : result;
}

infiniStatus_t Descriptor::Opaque::selectHygonLt(const MatmulInfo &info,
                                                const h3_lt::Config &requested_config, int expected_device) {
    auto config = requested_config;
    // DTK cuBLASLt compatibility lacks MIN_ALIGNMENT preference attributes.
    // Keep its default 256-byte contract and screen exact padded layouts out.
    config.alignment = 256;
    if (info.a_matrix.ld() % 128 || info.b_matrix.ld() % 128 || info.c_matrix.ld() % 128) {
        h3LtReceipt(config, "", "error", "dtk_lt_stride_alignment", -1, 0, 0);
        return INFINI_STATUS_NOT_IMPLEMENTED;
    }
    int current_device = -1, runtime_version = 0, blas_version = 0;
    cudaDeviceProp properties{};
    if (cudaGetDevice(&current_device) != cudaSuccess
        || cudaGetDeviceProperties(&properties, current_device) != cudaSuccess
        || cudaRuntimeGetVersion(&runtime_version) != cudaSuccess) {
        h3LtReceipt(config, "", "error", "device_identity", -1, 0, 0);
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    if (current_device != expected_device) {
        h3LtReceipt(config, "", "error", "device_mismatch", -1, 0, 0);
        return INFINI_STATUS_BAD_PARAM;
    }
    const auto version_status = internal->useCublas(nullptr, [&](cublasHandle_t handle) {
        return cublasGetVersion(handle, &blas_version) == CUBLAS_STATUS_SUCCESS
            ? INFINI_STATUS_SUCCESS : INFINI_STATUS_INTERNAL_ERROR;
    });
    if (version_status != INFINI_STATUS_SUCCESS) {
        h3LtReceipt(config, "", "error", "blas_identity", -1, 0, 0);
        return version_status;
    }
    std::ostringstream key;
    key << "h3lt1:" << config.runtime_tag << ":rt" << runtime_version
        << ":blas" << blas_version << ":d" << current_device << ':'
        << h3DeviceToken(properties.name) << ':' << properties.major << '_' << properties.minor
        << ":bf16_fp32:"
        << (info.a_matrix.row_stride == 1 ? 'N' : 'T')
        << (info.b_matrix.row_stride == 1 ? 'N' : 'T')
        << ':' << info.m << ':' << info.n << ':' << info.k << ':' << info.batch
        << ':' << info.a_matrix.ld() << ':' << info.b_matrix.ld() << ':' << info.c_matrix.ld()
        << ':' << info.a_matrix.stride << ':' << info.b_matrix.stride << ':' << info.c_matrix.stride
        << ":w" << config.workspace_limit << ":a" << config.alignment;
    lt_key = key.str();

    int choice = 0;
    bool matched = false;
    std::string error;
    if (!h3_lt::tableChoice(config, lt_key, choice, matched, error)) {
        h3LtReceipt(config, lt_key, "error", error.c_str(), -1, 0, 0);
        return INFINI_STATUS_BAD_PARAM;
    }
    if (!config.table_path.empty() && config.table_only && !matched) {
        h3LtReceipt(config, lt_key, "blas", "table_miss", -1, 0, 0);
        return INFINI_STATUS_SUCCESS;
    }
    if (!createBf16LtDescriptors(info)) {
        h3LtReceipt(config, lt_key, "error", "lt_descriptor", choice, 0, 0);
        return INFINI_STATUS_NOT_IMPLEMENTED;
    }
    cublasLtMatmulPreference_t preference = nullptr;
    if (cublasLtMatmulPreferenceCreate(&preference) != CUBLAS_STATUS_SUCCESS) {
        h3LtReceipt(config, lt_key, "error", "lt_preference", choice, 0, 0);
        return INFINI_STATUS_NOT_IMPLEMENTED;
    }
    bool ok = cublasLtMatmulPreferenceSetAttribute(
                  preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                  &config.workspace_limit, sizeof(config.workspace_limit)) == CUBLAS_STATUS_SUCCESS;
    cublasLtMatmulHeuristicResult_t candidates[h3_lt::kMaxCandidates]{};
    int found = 0;
    if (ok) {
        ok = cublasLtMatmulAlgoGetHeuristic(lt_handle, lt_desc, a_layout, b_layout,
                 c_layout, c_layout, preference, h3_lt::kMaxCandidates,
                 candidates, &found) == CUBLAS_STATUS_SUCCESS;
    }
    cublasLtMatmulPreferenceDestroy(preference);
    if (!ok || choice < 0 || choice >= found || found > static_cast<int>(h3_lt::kMaxCandidates)
        || candidates[choice].state != CUBLAS_STATUS_SUCCESS
        || candidates[choice].workspaceSize > config.workspace_limit) {
        h3LtReceipt(config, lt_key, "error", "heuristic_unavailable", choice, found, 0);
        return INFINI_STATUS_NOT_IMPLEMENTED;
    }
    lt_algo = candidates[choice].algo; // Cache an explicit algorithm per descriptor.
    lt_workspace = candidates[choice].workspaceSize;
    lt_alignment = config.alignment;
    lt_has_algo = true;
    h3LtReceipt(config, lt_key, "lt", matched ? "table" : "heuristic",
                choice, found, lt_workspace);
    return INFINI_STATUS_SUCCESS;
}
#endif

Descriptor::~Descriptor() {
#if !defined(ENABLE_ILUVATAR_API)
    if (_opaque) {
        _opaque->destroyLtDescriptors();
    }
#endif
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = c_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = MatmulInfo::create(c_desc, a_desc, b_desc, MatrixLayout::COL_MAJOR);
    CHECK_RESULT(result);

    auto info = result.take();
    auto opaque = new Opaque{handle->internal()};
    size_t workspace_bytes = 0;
#ifdef ENABLE_HYGON_API
    h3_lt::Config config;
    std::string error;
    if (!h3_lt::Config::read(config, error)) {
        h3LtReceipt(config, "", "error", error.c_str(), -1, 0, 0);
        delete opaque;
        return INFINI_STATUS_BAD_PARAM;
    }
    if (config.trace) {
        std::fprintf(stderr,
            "H3_GEMM_SHAPE {\"dtype\":%d,\"m\":%zu,\"n\":%zu,\"k\":%zu,"
            "\"batch\":%zu,\"transposed\":%d,\"op_a\":\"%c\",\"op_b\":\"%c\","
            "\"lda\":%td,\"ldb\":%td,\"ldc\":%td}\n",
            static_cast<int>(dtype), info.m, info.n, info.k, info.batch,
            info.is_transed ? 1 : 0,
            info.a_matrix.row_stride == 1 ? 'N' : 'T',
            info.b_matrix.row_stride == 1 ? 'N' : 'T',
            info.a_matrix.ld(), info.b_matrix.ld(), info.c_matrix.ld());
    }
    if (dtype == INFINI_DTYPE_BF16 && config.eligible(info.m, info.n, info.k, info.batch)) {
        if (info.a_matrix.ld() <= 0 || info.b_matrix.ld() <= 0 || info.c_matrix.ld() <= 0
            || info.a_matrix.ld() > 2147483647 || info.b_matrix.ld() > 2147483647
            || info.c_matrix.ld() > 2147483647) {
            delete opaque;
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }
        const auto status = opaque->selectHygonLt(info, config, handle->device_id);
        if (status != INFINI_STATUS_SUCCESS) {
            opaque->destroyLtDescriptors();
            delete opaque;
            return status;
        }
        workspace_bytes = opaque->lt_workspace;
    } else {
        h3LtReceipt(config, "", "blas", config.enabled ? "shape_or_dtype" : "disabled", -1, 0, 0);
    }
#elif !defined(ENABLE_ILUVATAR_API)
    if (dtype == INFINI_DTYPE_BF16 && !opaque->createBf16LtDescriptors(info)) {
        opaque->destroyLtDescriptors();
    }
#endif

    *desc_ptr = new Descriptor(
        dtype, info, workspace_bytes,
        opaque,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *c,
    float beta,
    const void *a,
    const void *b,
    float alpha,
    void *stream) const {

    cudaDataType a_type, b_type, c_type;
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
    cudaDataType compute_type;
#else
    cublasComputeType_t compute_type;
#endif

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        a_type = b_type = c_type = CUDA_R_16F;
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
        compute_type = CUDA_R_32F;
#else
        compute_type = CUBLAS_COMPUTE_32F;
#endif
        break;
    case INFINI_DTYPE_BF16:
        a_type = b_type = c_type = CUDA_R_16BF;
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
        compute_type = CUDA_R_32F;
#else
        compute_type = CUBLAS_COMPUTE_32F;
#endif
        break;
    case INFINI_DTYPE_F32:
        a_type = b_type = c_type = CUDA_R_32F;
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
        compute_type = CUDA_R_32F;
#else
        compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
#endif
        break;

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    if (_info.is_transed) {
        std::swap(a, b);
    }

    auto op_a = _info.a_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;
    auto op_b = _info.b_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;

#ifdef ENABLE_HYGON_API
    if (_opaque->lt_has_algo) {
        // Preflight must finish before Lt touches C; an executed Lt failure must
        // never fall back and apply beta again to a potentially modified C.
        if (!a || !b || !c) {
            return INFINI_STATUS_NULL_POINTER;
        }
        if (workspace_size < _opaque->lt_workspace
            || (_opaque->lt_workspace && !workspace)) {
            return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
        }
        const auto aligned = [](const void *p, size_t alignment) {
            return reinterpret_cast<uintptr_t>(p) % alignment == 0;
        };
        if (!aligned(a, _opaque->lt_alignment) || !aligned(b, _opaque->lt_alignment)
            || !aligned(c, _opaque->lt_alignment)
            || (_opaque->lt_workspace && !aligned(workspace, 256))) {
            return INFINI_STATUS_BAD_PARAM;
        }
        const auto status = cublasLtMatmul(
            _opaque->lt_handle, _opaque->lt_desc, &alpha,
            a, _opaque->a_layout, b, _opaque->b_layout, &beta,
            c, _opaque->c_layout, c, _opaque->c_layout, &_opaque->lt_algo,
            workspace, _opaque->lt_workspace, (cudaStream_t)stream);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "H3_GEMM_LT_EXECUTION_FAILED status=%d key=%s\n",
                         static_cast<int>(status), _opaque->lt_key.c_str());
            return INFINI_STATUS_INTERNAL_ERROR;
        }
        return INFINI_STATUS_SUCCESS;
    }
#elif !defined(ENABLE_ILUVATAR_API)
    if (_dtype == INFINI_DTYPE_BF16 && _opaque->lt_handle && _opaque->lt_desc
        && _opaque->a_layout && _opaque->b_layout && _opaque->c_layout) {
        auto lt_status = cublasLtMatmul(
            _opaque->lt_handle,
            _opaque->lt_desc,
            &alpha,
            a,
            _opaque->a_layout,
            b,
            _opaque->b_layout,
            &beta,
            c,
            _opaque->c_layout,
            c,
            _opaque->c_layout,
            nullptr,
            workspace,
            workspace_size,
            (cudaStream_t)stream);
        if (lt_status == CUBLAS_STATUS_SUCCESS) {
            return INFINI_STATUS_SUCCESS;
        }
    }
#endif

    CHECK_STATUS(_opaque->internal->useCublas(
        (cudaStream_t)stream,
        [&](cublasHandle_t handle) {
            CHECK_CUBLAS(
                cublasGemmStridedBatchedEx(
                    handle,
                    op_a,
                    op_b,
                    static_cast<int>(_info.m),
                    static_cast<int>(_info.n),
                    static_cast<int>(_info.k),
                    &alpha,
                    a,
                    a_type,
                    static_cast<int>(_info.a_matrix.ld()),
                    _info.a_matrix.stride,
                    b,
                    b_type,
                    static_cast<int>(_info.b_matrix.ld()),
                    _info.b_matrix.stride,
                    &beta,
                    c,
                    c_type,
                    static_cast<int>(_info.c_matrix.ld()),
                    _info.c_matrix.stride,
                    static_cast<int>(_info.batch),
                    compute_type,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            return INFINI_STATUS_SUCCESS;
        }));
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::gemm::nvidia
