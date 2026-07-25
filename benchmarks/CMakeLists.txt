# Scheduler benchmarks run against the mock engine so the numbers isolate
# scheduling cost from model compute. Real-model throughput is measured end to end
# by benchmarks/bench_batching.py against a running server instead.
add_executable(microvllm_bench
    bench_scheduler.cpp
)
target_link_libraries(microvllm_bench PRIVATE
    microvllm_core
    benchmark::benchmark_main
)
