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
  arches
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE RELEASE)

add_library(integral_indexing_utils SHARED)
target_sources(integral_indexing_utils PRIVATE ${PROJECT_SOURCE_DIR}/arches/src/integral_indexing_utils.cpp)
target_include_directories(integral_indexing_utils PRIVATE ${PROJECT_SOURCE_DIR}/arches/src/include)
target_compile_options(integral_indexing_utils PRIVATE -fPIC -Wall)
set_target_properties(integral_indexing_utils PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/arches/build)

add_library(integral_types SHARED)
target_sources(integral_types PRIVATE ${PROJECT_SOURCE_DIR}/arches/src/integral_types.cpp)
target_include_directories(integral_types PRIVATE ${PROJECT_SOURCE_DIR}/arches/src/include)
target_link_libraries(integral_types integral_indexing_utils)
target_compile_options(integral_types PRIVATE -fPIC -Wall)
set_target_properties(integral_types PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/arches/build)

add_library(matrix SHARED)
target_sources(matrix PRIVATE ${PROJECT_SOURCE_DIR}/arches/src/matrix.cpp)
target_include_directories(matrix PRIVATE ${PROJECT_SOURCE_DIR}/arches/src/include)
target_compile_options(matrix PRIVATE -fPIC -Wall)
set_target_properties(matrix PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/arches/build)

add_library(interface SHARED)
target_sources(interface PRIVATE ${PROJECT_SOURCE_DIR}/arches/src/array_interface.cpp)
target_include_directories(interface PRIVATE ${PROJECT_SOURCE_DIR}/arches/src/include)
target_compile_options(interface PRIVATE -fPIC -Wall)
set_target_properties(interface PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/arches/build)

add_library(jchunk SHARED)
target_sources(jchunk PRIVATE ${PROJECT_SOURCE_DIR}/arches/src/integrals.cpp)
target_include_directories(jchunk PRIVATE ${PROJECT_SOURCE_DIR}/arches/src/include)
target_compile_options(jchunk PRIVATE -fPIC -Wall)
set_target_properties(jchunk PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/arches/build)

add_library(kernels SHARED)
target_sources(kernels PRIVATE ${PROJECT_SOURCE_DIR}/arches/src/kernels.cpp)
target_include_directories(kernels PRIVATE ${PROJECT_SOURCE_DIR}/arches/src/include)
target_compile_options(kernels PRIVATE -fPIC -Wall)
set_target_properties(kernels PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/arches/build)

if(ARCHES_ENABLE_PYTHON)
    find_package (Python COMPONENTS Interpreter Development)
    add_library(arches_kernels SHARED)
    target_sources(arches_kernels PRIVATE ${PROJECT_SOURCE_DIR}/arches/src/integral_indexing_utils.cpp)
    target_include_directories(arches_kernels PRIVATE ${PROJECT_SOURCE_DIR}/arches/src)
    target_link_libraries(arches_kernels integral_indexing_utils ${Python_LIBRARIES})
    set_target_properties(arches_kernels
                            PROPERTIES
                            PREFIX ""
                            OUTPUT_NAME "kernels"
                            LINKER_LANGUAGE CXX)

endif(ARCHES_ENABLE_PYTHON)