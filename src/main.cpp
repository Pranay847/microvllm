#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include "microvllm/mock_engine.hpp"
#include "microvllm/model_engine.hpp"
#include "microvllm/server.hpp"
#include "microvllm/version.hpp"

#if MICROVLLM_HAS_LLAMA
#include "microvllm/llama_engine.hpp"
#endif

namespace {

using microvllm::IModelEngine;

struct Args {
    std::string  model;
    std::string  host    = "0.0.0.0";
    int          port    = 8080;
    int          threads = 8;      // Phase 0 operating point
    std::uint32_t n_ctx  = 4096;
    std::string  mock;             // if set, run the mock engine with this response
    bool         has_mock  = false;
    bool         mock_echo = false;  // mock echoes each prompt (for load-test matching)
    bool         quiet     = false;
    std::size_t  queue_depth = 64;
    std::size_t  batch_size  = 8;
    std::size_t  prefill_chunk = 128;
    std::uint32_t kv_blocks    = 0;
    std::uint32_t block_size   = 16;
    bool          no_prefix_cache = false;
    bool          log_requests    = false;
    microvllm::BatchingMode mode = microvllm::BatchingMode::kContinuous;
};

[[noreturn]] void usage(const char* prog, int code) {
    std::fprintf(code == 0 ? stdout : stderr,
                 "microvllm %s\n"
                 "usage: %s --model <path.gguf> [options]\n"
                 "       %s --mock <response> [options]   (no model; echoes a fixed response)\n"
                 "\n"
                 "options:\n"
                 "  --model <path>    GGUF model to serve\n"
                 "  --host <addr>     bind address (default 0.0.0.0)\n"
                 "  --port <n>        bind port (default 8080)\n"
                 "  --threads <n>     inference threads (default 8)\n"
                 "  --ctx <n>         per-request context length (default 4096)\n"
                 "  --mock <text>     serve a deterministic mock engine instead of a model\n"
                 "  --mock-echo       mock mode that echoes each prompt back (load testing)\n"
                 "  --queue <n>       max waiting requests before 503 (default 64)\n"
                 "  --batch-size <n>  sequences batched into one forward pass (default 8)\n"
                 "  --scheduler <m>   continuous (default) | static\n"
                 "  --prefill-chunk <n>  prompt tokens per sequence per step (default 128,\n"
                 "                    0 = submit whole prompts; continuous mode only)\n"
                 "  --kv-blocks <n>   KV-cache budget in blocks (default: derived from --ctx).\n"
                 "                    Lower it to exercise admission control and preemption\n"
                 "  --block-size <n>  tokens per KV block (default 16)\n"
                 "  --no-prefix-cache disable KV sharing between common prompt prefixes\n"
                 "  --quiet           silence llama.cpp info logging\n"
                 "  --help\n",
                 microvllm::kVersion, prog, prog);
    std::exit(code);
}

const char* need_value(int argc, char** argv, int& i, const char* prog) {
    if (i + 1 >= argc) {
        std::fprintf(stderr, "error: %s needs a value\n", argv[i]);
        usage(prog, 2);
    }
    return argv[++i];
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (std::strcmp(arg, "--model") == 0) {
            a.model = need_value(argc, argv, i, argv[0]);
        } else if (std::strcmp(arg, "--host") == 0) {
            a.host = need_value(argc, argv, i, argv[0]);
        } else if (std::strcmp(arg, "--port") == 0) {
            a.port = std::atoi(need_value(argc, argv, i, argv[0]));
        } else if (std::strcmp(arg, "--threads") == 0) {
            a.threads = std::atoi(need_value(argc, argv, i, argv[0]));
        } else if (std::strcmp(arg, "--ctx") == 0) {
            a.n_ctx = static_cast<std::uint32_t>(std::atoi(need_value(argc, argv, i, argv[0])));
        } else if (std::strcmp(arg, "--mock") == 0) {
            a.mock     = need_value(argc, argv, i, argv[0]);
            a.has_mock = true;
        } else if (std::strcmp(arg, "--mock-echo") == 0) {
            a.mock_echo = true;
            a.has_mock  = true;
        } else if (std::strcmp(arg, "--queue") == 0) {
            a.queue_depth = static_cast<std::size_t>(std::atoi(need_value(argc, argv, i, argv[0])));
        } else if (std::strcmp(arg, "--batch-size") == 0) {
            a.batch_size = static_cast<std::size_t>(std::atoi(need_value(argc, argv, i, argv[0])));
            if (a.batch_size == 0) {
                std::fprintf(stderr, "error: --batch-size must be >= 1\n");
                usage(argv[0], 2);
            }
        } else if (std::strcmp(arg, "--scheduler") == 0) {
            const char* m = need_value(argc, argv, i, argv[0]);
            if (std::strcmp(m, "continuous") == 0) {
                a.mode = microvllm::BatchingMode::kContinuous;
            } else if (std::strcmp(m, "static") == 0) {
                a.mode = microvllm::BatchingMode::kStatic;
            } else {
                std::fprintf(stderr, "error: --scheduler must be 'continuous' or 'static'\n");
                usage(argv[0], 2);
            }
        } else if (std::strcmp(arg, "--prefill-chunk") == 0) {
            a.prefill_chunk =
                static_cast<std::size_t>(std::atoi(need_value(argc, argv, i, argv[0])));
        } else if (std::strcmp(arg, "--kv-blocks") == 0) {
            a.kv_blocks =
                static_cast<std::uint32_t>(std::atoi(need_value(argc, argv, i, argv[0])));
        } else if (std::strcmp(arg, "--block-size") == 0) {
            a.block_size =
                static_cast<std::uint32_t>(std::atoi(need_value(argc, argv, i, argv[0])));
        } else if (std::strcmp(arg, "--no-prefix-cache") == 0) {
            a.no_prefix_cache = true;
        } else if (std::strcmp(arg, "--log-requests") == 0) {
            a.log_requests = true;
        } else if (std::strcmp(arg, "--quiet") == 0) {
            a.quiet = true;
        } else if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            usage(argv[0], 0);
        } else {
            std::fprintf(stderr, "error: unknown argument '%s'\n", arg);
            usage(argv[0], 2);
        }
    }
    if (!a.has_mock && a.model.empty()) {
        std::fprintf(stderr, "error: one of --model or --mock is required\n");
        usage(argv[0], 2);
    }
    return a;
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);
    const microvllm::ServerConfig server_cfg{.host            = args.host,
                                             .port            = args.port,
                                             .max_queue_depth = args.queue_depth,
                                             .max_batch_size  = args.batch_size,
                                             .mode            = args.mode,
                                             .prefill_chunk   = args.prefill_chunk,
                                             .kv_blocks       = args.kv_blocks,
                                             .block_size      = args.block_size,
                                             .prefix_caching  = !args.no_prefix_cache,
                                             .log_requests    = args.log_requests};

    if (args.has_mock) {
        std::printf("microvllm %s (mock engine%s)\n", microvllm::kVersion,
                    args.mock_echo ? ", echo" : "");
        microvllm::MockModelEngine engine(microvllm::MockModelEngine::Config{
            .response = args.mock, .echo_prompt = args.mock_echo});
        return microvllm::serve(engine, server_cfg) ? 0 : 1;
    }

#if MICROVLLM_HAS_LLAMA
    if (args.quiet) {
        microvllm::LlamaBackend::quiet_logging();
    }
    const microvllm::LlamaBackend backend;
    std::printf("microvllm %s | %s\n", microvllm::kVersion, microvllm::LlamaBackend::system_info());

    try {
        microvllm::LlamaEngineConfig cfg;
        cfg.model_path = args.model;
        cfg.n_ctx      = args.n_ctx;
        cfg.n_threads  = args.threads;
        // The context must be built to hold as many sequences as the scheduler will
        // batch; otherwise llama.cpp aborts the process on the first oversized batch.
        cfg.n_seq_max  = static_cast<std::uint32_t>(
            std::max<std::size_t>(args.batch_size, cfg.n_seq_max));
        // llama.cpp divides the KV pool across sequences, so its n_ctx is a TOTAL and
        // each sequence gets n_ctx / n_seq_max. --ctx means what a user expects it to
        // mean -- the context available to one request -- so scale it up here. Without
        // this, raising --batch-size silently shrinks every request's usable context.
        cfg.n_ctx      = args.n_ctx * cfg.n_seq_max;
        microvllm::LlamaModelEngine engine(cfg);
        return microvllm::serve(engine, server_cfg) ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return 1;
    }
#else
    std::fprintf(stderr, "error: built without llama.cpp; only --mock is available\n");
    return 1;
#endif
}
