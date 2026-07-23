# Scheduler benchmarks run against the mock engine so the numbers isolate
# scheduling cost from model compute. Real-model benchmarks are driven from
# scripts/run_bench_suite.sh against the running server instead.
add_executable(microvllm_bench
    bench_smoke.cpp
)
target_link_libraries(microvllm_bench PRIVATE
    microvllm_core
    benchmark::benchmark_main
)
