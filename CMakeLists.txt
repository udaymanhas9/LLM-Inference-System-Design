cmake_minimum_required(VERSION 3.15)
project(fastino_inference CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

# ── llama.cpp (git submodule at third_party/llama.cpp) ───────────────────────
#
# First-time setup:
#   git submodule add https://github.com/ggml-org/llama.cpp third_party/llama.cpp
#   git submodule update --init --recursive
#
# Metal (Apple GPU) is enabled automatically on macOS when building with Xcode
# or a recent CMake that detects the Metal framework.
#
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt")
    message(FATAL_ERROR
        "llama.cpp submodule missing.\n"
        "Run:  git submodule add https://github.com/ggml-org/llama.cpp third_party/llama.cpp\n"
        "      git submodule update --init --recursive"
    )
endif()

set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)

add_subdirectory(third_party/llama.cpp)

# ── Our inference shared library ─────────────────────────────────────────────
add_subdirectory(src/cpp)
