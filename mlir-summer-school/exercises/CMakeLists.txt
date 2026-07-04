# The MLIR summer school exercise project: a tiny out-of-tree MLIR project
# with one dialect (`school`), three passes, and lit/FileCheck tests.
#
# Configure against a prebuilt MLIR:
#   cmake -G Ninja -S . -B build -DMLIR_DIR=<mlir-build-or-install>/lib/cmake/mlir
#
# See README.md for the full setup instructions.
cmake_minimum_required(VERSION 3.20.0)
project(mlir-summer-school LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)

# Default arguments passed to `lit` when running the tests.
set(LLVM_LIT_ARGS "-sv" CACHE STRING "Default options for lit")

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

set(SCHOOL_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(SCHOOL_BINARY_DIR ${PROJECT_BINARY_DIR})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${SCHOOL_SOURCE_DIR}/include)
include_directories(${SCHOOL_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include/School)
add_subdirectory(lib/School)
add_subdirectory(school-opt)
add_subdirectory(test)
