# First check if Intel or AdaptiveCpp compiler exists. If so, use them.
find_package(IntelSYCL QUIET)
if (IntelSYCL_FOUND)
  set(HAS_SYCL TRUE)
  return()
endif()

find_package(AdaptiveCpp QUIET)
if(AdaptiveCpp_FOUND)
  set(HAS_SYCL TRUE)
  return()
endif()

# If neither is found, let's check if the current CMAKE_CXX_COMPILER
# supports "-fsycl" flag.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-fsycl" HAS_SYCL)

# Couldn't find a compiler which supports SYCL. Let's return.
if (NOT HAS_SYCL)
  return()
endif()

# Compiler supports SYCL, let's set SYCL_FLAGS.
set(SYCL_FLAGS "-fsycl")

# Let's check which backends it supports. Currently, only checks for
# CUDA.
check_cxx_compiler_flag("${SYCL_FLAGS} -fsycl-targets=nvptx64-nvidia-cuda"
  HAS_SYCL_CUDA)
if (NOT HAS_SYCL_CUDA)
  return()
endif()
set(SYCL_FLAGS "${SYCL_FLAGS} -fsycl-targets=nvptx64-nvidia-cuda")

# Let's try to figure out the CUDA GPU architecture now. This doesn't
# work now as any architecture is accepted whether the hardware/runtime
# supports it or not.

#set(CUDA_ARCHS 90 80 70 60)
#foreach(ARCH ${CUDA_ARCHS})
#  check_cxx_compiler_flag("${SYCL_FLAGS} --cuda-gpu-arch=sm_${ARCH}"
#    HAS_SYCL_CUDA_SM${ARCH})
#  if (HAS_SYCL_CUDA_SM${ARCH})
#    set(SYCL_FLAGS "${SYCL_FLAGS} --cuda-gpu-arch=sm_${ARCH}")
#    return()
#  endif()
#endforeach()

if (COMMAND add_sycl_to_target)
  return()
endif()

function(add_sycl_to_target)
  set(options)
  set(one_value_keywords TARGET)
	# We don't use SOURCE but provide the following for compatibility.
  set(multi_value_keywords SOURCES)
  cmake_parse_arguments(ADD_SYCL
    "${options}"
    "${one_value_keywords}"
    "${multi_value_keywords}"
    ${ARGN}
  )

  separate_arguments(SYCL_FLAGS UNIX_COMMAND "${SYCL_FLAGS}")
  target_compile_options(${ADD_SYCL_TARGET} PRIVATE ${SYCL_FLAGS})
  target_link_options(${ADD_SYCL_TARGET} PRIVATE ${SYCL_FLAGS})
endfunction()
