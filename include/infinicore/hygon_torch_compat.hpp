#pragma once

// The cluster PyTorch wheel was built with gflags/glog/NUMA enabled, but its
// conda prefix only ships the runtime library and not those development
// headers. InfiniCore does not use any of their APIs, so disable the optional
// integrations before any other PyTorch/C10 header is parsed.
#include <c10/macros/cmake_macros.h>

#undef C10_USE_GLOG
#undef C10_USE_GFLAGS
#undef C10_USE_NUMA
