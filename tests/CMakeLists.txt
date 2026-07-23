# Tests link microvllm_core ONLY -- never llama.cpp. That keeps the suite fast
# enough to run on every push, removes the 469 MB model from CI entirely, and
# means a ThreadSanitizer report can only be about first-party code.
add_executable(microvllm_tests
    test_types.cpp
)
target_link_libraries(microvllm_tests PRIVATE
    microvllm_core
    GTest::gtest_main
)

gtest_discover_tests(microvllm_tests
    PROPERTIES LABELS "unit"
    DISCOVERY_TIMEOUT 60  # sanitizer builds start slowly
)
