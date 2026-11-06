#pragma once

// The cluster PyTorch wheel was built with gflags/glog/NUMA enabled, but its
// conda prefix only ships the runtime library and not those development
// headers. InfiniCore does not use any of their APIs, so disable the optional
// integrations before any other PyTorch/C10 header is parsed.
//
// Some PyTorch wheels do not ship c10/macros/cmake_macros.h.  The header is
// only needed when the wheel provides it; on wheels without it the following
// #undef operations are harmless and the build can continue.
#if defined(__has_include)
#  if __has_include(<c10/macros/cmake_macros.h>)
#    include <c10/macros/cmake_macros.h>
#  endif
#else
#  include <c10/macros/cmake_macros.h>
#endif

#undef C10_USE_GLOG
#undef C10_USE_GFLAGS
#undef C10_USE_NUMA
