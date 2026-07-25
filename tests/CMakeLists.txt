# Tests link microvllm_core ONLY -- never llama.cpp. That keeps the suite fast
# enough to run on every push, removes the 469 MB model from CI entirely, and
# means a ThreadSanitizer report can only be about first-party code.
add_executable(microvllm_tests
    test_types.cpp
    test_mock_engine.cpp
    test_generator.cpp
    test_http_api.cpp
    test_request_queue.cpp
    test_scheduler.cpp
    test_scheduler_serial.cpp
    test_concurrency.cpp
)
target_link_libraries(microvllm_tests PRIVATE
    microvllm_core
    microvllm_http_api
    GTest::gtest_main
)

# ---------------------------------------------------------------------------
# ThreadSanitizer vs. Ubuntu 24.04 ASLR.
#
# The kernel ships vm.mmap_rnd_bits=32, which is more entropy than TSan's shadow
# memory mapping tolerates. Every TSan binary aborts before main() with:
#
#     FATAL: ThreadSanitizer: unexpected memory mapping 0x...
#
# GitHub Actions' ubuntu-24.04 runners have the same setting, so this must be
# fixed in the build rather than by hand. `setarch -R` disables ASLR for the
# child process and needs no privileges -- unlike the usual `sysctl
# vm.mmap_rnd_bits=28`, which requires root and does not survive a reboot.
#
# Disabling ASLR does not weaken the result: TSan detects races via happens-before
# tracking, which is independent of where the address space happens to land.
#
# CROSSCOMPILING_EMULATOR is what gtest_discover_tests uses as its test executor,
# so setting it covers both build-time discovery and ctest execution.
# ---------------------------------------------------------------------------
if(MICROVLLM_SANITIZER STREQUAL "thread")
    find_program(MICROVLLM_SETARCH setarch)
    if(MICROVLLM_SETARCH)
        set_target_properties(microvllm_tests PROPERTIES
            CROSSCOMPILING_EMULATOR "${MICROVLLM_SETARCH};-R")
        message(STATUS "microvllm: TSan tests run under '${MICROVLLM_SETARCH} -R' (ASLR disabled)")
    else()
        message(WARNING
            "setarch not found. ThreadSanitizer tests will likely abort with "
            "'unexpected memory mapping'. Install util-linux, or run "
            "'sudo sysctl vm.mmap_rnd_bits=28' as a privileged fallback.")
    endif()
endif()

gtest_discover_tests(microvllm_tests
    PROPERTIES LABELS "unit"
    DISCOVERY_TIMEOUT 60  # sanitizer builds start slowly
)
