cmake_minimum_required(VERSION 3.22.1)
project(9_sycl_of_hell LANGUAGES CXX)

option(ENABLE_ASAN "Enable AddressSanitizer" OFF)

# Set CMake module path.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

# https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling
# Use, i.e. don't skip the full RPATH for the build tree.
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
# When building, don't use the install RPATH.
set(CMAKE_SKIP_BUILD_RPATH FALSE)
# Set the library location.
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
# Add the automatically determined parts of the RPATH which point to directories
# outside the build tree to the install RPATH.
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

include(FindSYCLTargets)

# Turn on AddressSanitizer if ENABLE_ASAN is set.
if (ENABLE_ASAN)
  include(CheckLinkerFlag)
  check_linker_flag(CXX -fsanitize=address HAS_ADDRESS_SANITIZER)
  if (NOT HAS_ADDRESS_SANITIZER)
    message(FATAL_ERROR "AddressSanitizer is enabled but not supported.")
  endif()
endif()

file(GLOB files "*.cpp")
foreach(file_name ${files})
  get_filename_component(example_name ${file_name} NAME_WE)
  add_executable(${example_name} ${file_name})
  target_include_directories(${example_name} PRIVATE ../3rd_party)
  add_sycl_to_target(TARGET ${example_name} SOURCES ${file_name})

  if (HAS_ADDRESS_SANITIZER)
    target_compile_options(${example_name} PRIVATE -fsanitize=address)
    target_link_options(${example_name} PRIVATE -fsanitize=address)
  endif()
  install(TARGETS ${example_name} RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
endforeach()

# Add clang-format as a custom target if available.
find_program(CLANG_FORMAT NAMES clang-format)
if (CLANG_FORMAT)
  add_custom_target(format
    COMMAND ${CLANG_FORMAT} -i ${CMAKE_CURRENT_SOURCE_DIR}/*.[ch]pp
    COMMENT "Running clang-format ...")
  add_custom_target(format-check
    COMMAND ${CLANG_FORMAT} --dry-run -Werror -i ${CMAKE_CURRENT_SOURCE_DIR}/*.[ch]pp
    COMMENT "Running clang-format check ...")
endif()

# Configure .clang-tidy.in to generate .clang-tidy.
configure_file(${CMAKE_SOURCE_DIR}/.clang-tidy.in ${CMAKE_SOURCE_DIR}/.clang-tidy @ONLY)
# Add clang-tidy as a custom target if available.
find_program(CLANG_TIDY NAMES clang-tidy)
if (CLANG_TIDY)
  add_custom_target(tidy
    COMMAND ${CLANG_TIDY} -p ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*.[ch]pp
    COMMENT "Running clang-tidy ...")
endif()
