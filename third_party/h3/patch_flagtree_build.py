"""Honor the upstream profiler-off option for the HCU runtime plugin."""
import os
from pathlib import Path

import sys
path = Path(sys.argv[1]) / 'cmake/FlagTreeOptions.cmake'
old = '''    list(APPEND TRITON_PLUGIN_DIRS
      "${CMAKE_CURRENT_SOURCE_DIR}/third_party/hcu/proton")'''
new = '''    if(TRITON_BUILD_PROTON)
      list(APPEND TRITON_PLUGIN_DIRS
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/hcu/proton")
    endif()'''
text = path.read_text()
if new not in text:
    if text.count(old) != 1:
        raise RuntimeError('Unexpected HCU profiler registration; refusing a broad replacement')
    path.write_text(text.replace(old, new, 1))
text = path.read_text()
anchor = '''      PRIVATE DistributedIR SIMTIR ProtonIR Python3::Module pybind11::headers)'''
dependencies = '''    add_dependencies(TritonDistributed
      TritonDistributedToLLVMConversionPassIncGen
      TritonDistributedConversionPassIncGen
      TritonConversionPassIncGen LLVMIRIncGen)'''
if dependencies not in text:
    if text.count(anchor) != 1:
        raise RuntimeError('Unexpected distributed plugin definition')
    path.write_text(text.replace(anchor, anchor + '\n' + dependencies, 1))
print('HCU_BUILD_PATCHES_READY', flush=True)

# Developer CLI tools are not required by the HCU Python compiler/runtime.
# Keep their targets available explicitly without linking them for every wheel.
hcu_cmake = path.parents[1] / 'third_party/hcu/CMakeLists.txt'
source = hcu_cmake.read_text()
source = source.replace('add_subdirectory(bin)', 'add_subdirectory(bin EXCLUDE_FROM_ALL)')
hcu_cmake.write_text(source)
print('HCU_DEVELOPER_TOOLS_EXCLUDED_FROM_WHEEL_BUILD', flush=True)
