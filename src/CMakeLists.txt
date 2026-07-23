# ---------------------------------------------------------------------------
# microvllm_core -- scheduler, queue, block allocator, metrics, mock engine.
#
# Deliberately has NO dependency on llama.cpp. This is what makes the project
# testable: unit tests and the ThreadSanitizer target link only this, so CI runs
# the full scheduling path in milliseconds without downloading a 469 MB model,
# and TSan reports can only come from first-party code.
# ---------------------------------------------------------------------------
add_library(microvllm_core STATIC
    core/types.cpp
)
target_link_libraries(microvllm_core PUBLIC microvllm_options)
add_library(microvllm::core ALIAS microvllm_core)

# ---------------------------------------------------------------------------
# microvllm_llama -- the only translation units allowed to include llama.h.
# ---------------------------------------------------------------------------
if(MICROVLLM_BUILD_LLAMA)
    # Upstream renamed this target from `common` to `llama-common`. CMake silently
    # degrades an unknown target name into a raw `-l<name>` link flag, so without
    # this check a rename surfaces as a cryptic "cannot find -lcommon" at link time.
    foreach(_dep llama llama-common)
        if(NOT TARGET ${_dep})
            message(FATAL_ERROR
                "Expected llama.cpp to define target '${_dep}'. The pinned submodule "
                "(third_party/llama.cpp) may have renamed it. Check its CMakeLists.")
        endif()
    endforeach()

    add_library(microvllm_llama STATIC
        engine/llama_engine.cpp
    )
    target_link_libraries(microvllm_llama
        PUBLIC  microvllm_core
        PRIVATE llama llama-common)
    add_library(microvllm::llama ALIAS microvllm_llama)
endif()

# ---------------------------------------------------------------------------
# Server executable
# ---------------------------------------------------------------------------
add_executable(microvllm main.cpp)
target_link_libraries(microvllm PRIVATE
    microvllm_core
    nlohmann_json::nlohmann_json
    httplib::httplib
)
if(MICROVLLM_BUILD_LLAMA)
    target_link_libraries(microvllm PRIVATE microvllm_llama)
    target_compile_definitions(microvllm PRIVATE MICROVLLM_HAS_LLAMA=1)
endif()
