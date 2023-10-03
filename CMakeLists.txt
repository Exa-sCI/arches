######################################################################
# CMake version and policies
######################################################################
cmake_minimum_required(VERSION 3.20.0)
# Note that cmake_minimum_required affects policy defaults.
# All policies known to the running version of CMake and introduced in
# cmake_minimum_required version or earlier will be set to use NEW behavior

if(POLICY CMP0127)
  cmake_policy(SET CMP0127 NEW) # Condition evaluation v3.22+ https://cmake.org/cmake/help/latest/policy/CMP0127.html
endif()

######################################################################
# Quantum Envelope
######################################################################
project(
  quanatum_envelope
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE RELEASE)

add_library(integral_indexing_utils SHARED)
target_sources(integral_indexing_utils PRIVATE ${PROJECT_SOURCE_DIR}/qe/src/integral_indexing_utils.cpp)
target_include_directories(integral_indexing_utils PRIVATE ${PROJECT_SOURCE_DIR}/qe/src)
target_compile_options(integral_indexing_utils PRIVATE -fPIC -Wall)
set_target_properties(integral_indexing_utils PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/qe/build)