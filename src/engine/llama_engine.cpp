#include "microvllm/llama_engine.hpp"

#include <llama.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace microvllm {

// ---------------------------------------------------------------------------
// LlamaBackend
// ---------------------------------------------------------------------------
LlamaBackend::LlamaBackend() {
    llama_backend_init();
}

LlamaBackend::~LlamaBackend() {
    llama_backend_free();
}

const char* LlamaBackend::system_info() noexcept {
    return llama_print_system_info();
}

void LlamaBackend::quiet_logging() noexcept {
    llama_log_set(
        [](ggml_log_level level, const char* text, void* /*user*/) {
            if (level >= GGML_LOG_LEVEL_WARN) {
                std::fputs(text, stderr);
            }
        },
        nullptr);
}

// ---------------------------------------------------------------------------
// RAII wrappers over the C handles
// ---------------------------------------------------------------------------
namespace {

struct ModelDeleter {
    void operator()(llama_model* m) const noexcept { llama_model_free(m); }
};
struct ContextDeleter {
    void operator()(llama_context* c) const noexcept { llama_free(c); }
};
struct SamplerDeleter {
    void operator()(llama_sampler* s) const noexcept { llama_sampler_free(s); }
};

using ModelPtr   = std::unique_ptr<llama_model, ModelDeleter>;
using ContextPtr = std::unique_ptr<llama_context, ContextDeleter>;
using SamplerPtr = std::unique_ptr<llama_sampler, SamplerDeleter>;

}  // namespace

// ---------------------------------------------------------------------------
// LlamaModelEngine::Impl
// ---------------------------------------------------------------------------
struct LlamaModelEngine::Impl {
    ModelPtr            model;
    ContextPtr          context;
    const llama_vocab*  vocab = nullptr;
    llama_memory_t      memory = nullptr;
    llama_batch         batch{};
    Token               eos = -1;
    std::uint32_t       n_batch = 0;
    std::unordered_map<SeqId, SamplerPtr> samplers;

    explicit Impl(const LlamaEngineConfig& cfg) {
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = cfg.n_gpu_layers;
        model.reset(llama_model_load_from_file(cfg.model_path.c_str(), mp));
        if (!model) {
            throw std::runtime_error("failed to load model: " + cfg.model_path);
        }
        vocab = llama_model_get_vocab(model.get());

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = cfg.n_ctx;
        cp.n_batch         = cfg.n_batch;
        cp.n_seq_max       = cfg.n_seq_max;
        cp.n_threads       = cfg.n_threads;
        cp.n_threads_batch = cfg.n_threads;
        context.reset(llama_init_from_model(model.get(), cp));
        if (!context) {
            throw std::runtime_error("failed to create llama_context");
        }

        memory  = llama_get_memory(context.get());
        eos     = llama_vocab_eos(vocab);
        n_batch = llama_n_batch(context.get());

        // One reusable batch. Third arg is max sequences a single token may belong to;
        // 1 until Phase 5 introduces prefix sharing.
        batch = llama_batch_init(static_cast<std::int32_t>(n_batch), 0, 1);
    }

    ~Impl() { llama_batch_free(batch); }

    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
};

// ---------------------------------------------------------------------------
// LlamaModelEngine
// ---------------------------------------------------------------------------
LlamaModelEngine::LlamaModelEngine(const LlamaEngineConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

LlamaModelEngine::~LlamaModelEngine() = default;

EngineCaps LlamaModelEngine::caps() const {
    return EngineCaps{
        .n_ctx     = llama_n_ctx(impl_->context.get()),
        .n_batch   = llama_n_batch(impl_->context.get()),
        .n_seq_max = llama_n_seq_max(impl_->context.get()),
        .eos       = impl_->eos,
    };
}

std::vector<Token> LlamaModelEngine::tokenize(std::string_view text, bool add_special) {
    // llama_tokenize returns the count written, or -(needed) if the buffer is too small.
    std::vector<Token> tokens(text.size() + 2);  // + room for a leading special token
    std::int32_t n = llama_tokenize(impl_->vocab, text.data(), static_cast<std::int32_t>(text.size()),
                                    tokens.data(), static_cast<std::int32_t>(tokens.size()),
                                    add_special, /*parse_special=*/true);
    if (n < 0) {
        tokens.resize(static_cast<std::size_t>(-n));
        n = llama_tokenize(impl_->vocab, text.data(), static_cast<std::int32_t>(text.size()),
                           tokens.data(), static_cast<std::int32_t>(tokens.size()),
                           add_special, /*parse_special=*/true);
    }
    tokens.resize(static_cast<std::size_t>(n < 0 ? 0 : n));
    return tokens;
}

std::string LlamaModelEngine::piece(Token token) {
    char buf[256];
    // special=false so control tokens (BOS/EOS/etc.) render as empty text.
    std::int32_t n = llama_token_to_piece(impl_->vocab, token, buf, sizeof(buf), 0, /*special=*/false);
    if (n >= 0) {
        return std::string(buf, static_cast<std::size_t>(n));
    }
    std::string s(static_cast<std::size_t>(-n), '\0');
    n = llama_token_to_piece(impl_->vocab, token, s.data(), static_cast<std::int32_t>(s.size()), 0, false);
    s.resize(static_cast<std::size_t>(n < 0 ? 0 : n));
    return s;
}

void LlamaModelEngine::begin_sequence(SeqId seq, const SamplingParams& p) {
    llama_sampler* chain = llama_sampler_chain_init(llama_sampler_chain_default_params());

    if (p.repeat_penalty != 1.0F) {
        llama_sampler_chain_add(
            chain, llama_sampler_init_penalties(/*last_n=*/64, p.repeat_penalty, /*freq=*/0.0F, /*present=*/0.0F));
    }
    if (p.temperature <= 0.0F) {
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    } else {
        if (p.top_k > 0) {
            llama_sampler_chain_add(chain, llama_sampler_init_top_k(p.top_k));
        }
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(p.top_p, /*min_keep=*/1));
        llama_sampler_chain_add(chain, llama_sampler_init_temp(p.temperature));
        const std::uint32_t seed = p.seed == 0 ? LLAMA_DEFAULT_SEED : p.seed;
        llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));
    }

    impl_->samplers[seq] = SamplerPtr(chain);
}

std::vector<GenStep> LlamaModelEngine::decode(std::span<const BatchItem> items) {
    llama_batch& b = impl_->batch;

    std::size_t total = 0;
    for (const BatchItem& item : items) {
        total += item.tokens.size();
    }
    if (total > impl_->n_batch) {
        throw std::runtime_error("decode: batch of " + std::to_string(total) +
                                 " tokens exceeds n_batch=" + std::to_string(impl_->n_batch) +
                                 " (chunked prefill lands in Phase 4)");
    }

    struct SampleAt {
        SeqId        seq;
        std::int32_t index;
    };
    std::vector<SampleAt> to_sample;

    b.n_tokens = 0;
    for (const BatchItem& item : items) {
        for (std::size_t k = 0; k < item.tokens.size(); ++k) {
            const std::int32_t i = b.n_tokens;
            b.token[i]     = item.tokens[k];
            b.pos[i]       = item.pos0 + static_cast<Pos>(k);
            b.n_seq_id[i]  = 1;
            b.seq_id[i][0] = item.seq;
            const bool is_last = (k + 1 == item.tokens.size());
            b.logits[i]    = static_cast<std::int8_t>(item.sample && is_last ? 1 : 0);
            ++b.n_tokens;
        }
        if (item.sample && !item.tokens.empty()) {
            to_sample.push_back({item.seq, b.n_tokens - 1});
        }
    }

    const std::int32_t rc = llama_decode(impl_->context.get(), b);
    if (rc != 0) {
        throw std::runtime_error("llama_decode failed with code " + std::to_string(rc));
    }

    std::vector<GenStep> steps;
    steps.reserve(to_sample.size());
    for (const SampleAt& s : to_sample) {
        auto it = impl_->samplers.find(s.seq);
        if (it == impl_->samplers.end()) {
            throw std::logic_error("decode on a sequence without begin_sequence");
        }
        const Token tok = llama_sampler_sample(it->second.get(), impl_->context.get(), s.index);
        steps.push_back(GenStep{
            .seq = s.seq, .token = tok, .is_eog = llama_vocab_is_eog(impl_->vocab, tok)});
    }
    return steps;
}

void LlamaModelEngine::release_sequence(SeqId seq) {
    impl_->samplers.erase(seq);                       // frees the sampler chain
    llama_memory_seq_rm(impl_->memory, seq, -1, -1);  // drop the whole sequence's KV cache
}

}  // namespace microvllm
