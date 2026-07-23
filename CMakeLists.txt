cmake_minimum_required(VERSION 3.25)  # add_subdirectory(... SYSTEM)

project(microvllm
    VERSION     0.1.0
    DESCRIPTION "A miniature LLM inference server: continuous batching + block-based KV cache"
    LANGUAGES   CXX)

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

option(MICROVLLM_BUILD_TESTS      "Build unit tests"                  ON)
option(MICROVLLM_BUILD_BENCHMARKS "Build benchmarks"                  ON)
option(MICROVLLM_BUILD_LLAMA      "Build the llama.cpp-backed engine" ON)

set(MICROVLLM_SANITIZER "none" CACHE STRING
    "Sanitizer applied to FIRST-PARTY targets only: none|address|thread|undefined|address+undefined")
set_property(CACHE MICROVLLM_SANITIZER PROPERTY STRINGS
    none address thread undefined address+undefined)

include(FetchContent)
include(CTest)

# ---------------------------------------------------------------------------
# Warnings and sanitizers, as interface targets.
#
# These are deliberately applied ONLY to first-party targets. llama.cpp and the
# fetched dependencies are consumed as-is: instrumenting ggml's thread pool
# produces a flood of TSan reports that say nothing about our scheduler, and
# building it with -O1 -fsanitize would make benchmarks meaningless.
# ---------------------------------------------------------------------------
add_library(microvllm_warnings INTERFACE)
if(MSVC)
    target_compile_options(microvllm_warnings INTERFACE /W4 /permissive- /Zc:__cplusplus)
else()
    target_compile_options(microvllm_warnings INTERFACE
        -Wall -Wextra -Wpedantic
        -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wcast-align
        -Wunused -Woverloaded-virtual -Wconversion -Wsign-conversion
        -Wdouble-promotion -Wformat=2)
endif()

add_library(microvllm_sanitizers INTERFACE)
if(NOT MICROVLLM_SANITIZER STREQUAL "none")
    if(MSVC)
        # MSVC has ASan only. No TSan exists on Windows at all -- this is why the
        # project's primary target is WSL2/Linux.
        if(NOT MICROVLLM_SANITIZER STREQUAL "address")
            message(FATAL_ERROR
                "MICROVLLM_SANITIZER='${MICROVLLM_SANITIZER}' is not available under MSVC. "
                "Only 'address' is supported on Windows; use a wsl-* preset for thread/undefined.")
        endif()
        target_compile_options(microvllm_sanitizers INTERFACE /fsanitize=address)
    else()
        if(MICROVLLM_SANITIZER STREQUAL "address+undefined")
            set(_san_flags -fsanitize=address -fsanitize=undefined -fno-sanitize-recover=all)
        else()
            set(_san_flags -fsanitize=${MICROVLLM_SANITIZER})
        endif()
        target_compile_options(microvllm_sanitizers INTERFACE
            ${_san_flags} -fno-omit-frame-pointer -g)
        target_link_options(microvllm_sanitizers INTERFACE ${_san_flags})
        unset(_san_flags)
    endif()
    message(STATUS "microvllm: sanitizer = ${MICROVLLM_SANITIZER} (first-party targets only)")
endif()

# Convenience: everything first-party links this.
add_library(microvllm_options INTERFACE)
target_link_libraries(microvllm_options INTERFACE microvllm_warnings microvllm_sanitizers)
target_include_directories(microvllm_options INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---------------------------------------------------------------------------
# llama.cpp (vendored submodule, pinned to a release tag)
# ---------------------------------------------------------------------------
if(MICROVLLM_BUILD_LLAMA)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt")
        message(FATAL_ERROR
            "third_party/llama.cpp is empty. Run: git submodule update --init --recursive")
    endif()

    # As a subdirectory LLAMA_STANDALONE is OFF, so tools/tests/examples already
    # default to OFF. Pinned explicitly so a llama.cpp default change cannot
    # silently add minutes to our build.
    set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)  # we use common's batch/sampler helpers
    set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_APP      OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_UI       OFF CACHE BOOL "" FORCE)
    set(LLAMA_OPENSSL        OFF CACHE BOOL "" FORCE)  # we do not use llama's HTTPS paths
    set(GGML_CCACHE          ON  CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)

    add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL SYSTEM)
endif()

# ---------------------------------------------------------------------------
# Third-party dependencies
# ---------------------------------------------------------------------------
FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.12.0
    GIT_SHALLOW    TRUE
    SYSTEM)

# Our own HTTP layer. llama.cpp vendors its own copy of cpp-httplib under a
# different target name (cpp-httplib vs httplib::httplib), so there is no clash.
FetchContent_Declare(httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        v0.50.1
    GIT_SHALLOW    TRUE
    SYSTEM)

FetchContent_MakeAvailable(nlohmann_json httplib)

if(MICROVLLM_BUILD_TESTS)
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.17.0
        GIT_SHALLOW    TRUE
        SYSTEM)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    include(GoogleTest)
endif()

if(MICROVLLM_BUILD_BENCHMARKS)
    FetchContent_Declare(benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.9.5
        GIT_SHALLOW    TRUE
        SYSTEM)
    set(BENCHMARK_ENABLE_TESTING     OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL     OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)
endif()

# ---------------------------------------------------------------------------
# First-party targets
# ---------------------------------------------------------------------------
add_subdirectory(src)

if(MICROVLLM_BUILD_TESTS AND BUILD_TESTING)
    add_subdirectory(tests)
endif()

if(MICROVLLM_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "microvllm ${PROJECT_VERSION}")
message(STATUS "  build type ......... ${CMAKE_BUILD_TYPE}")
message(STATUS "  compiler ........... ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  sanitizer .......... ${MICROVLLM_SANITIZER}")
message(STATUS "  llama.cpp engine ... ${MICROVLLM_BUILD_LLAMA}")
message(STATUS "  tests .............. ${MICROVLLM_BUILD_TESTS}")
message(STATUS "  benchmarks ......... ${MICROVLLM_BUILD_BENCHMARKS}")
message(STATUS "")
