#include "microvllm/http_api.hpp"

#include <nlohmann/json.hpp>

namespace microvllm {

using nlohmann::json;

RequestSpec parse_generate_request(std::string_view body) {
    json j;
    try {
        j = json::parse(body);
    } catch (const json::parse_error& e) {
        throw BadRequest(std::string("invalid JSON: ") + e.what());
    }
    if (!j.is_object()) {
        throw BadRequest("request body must be a JSON object");
    }

    RequestSpec spec;

    // Required: prompt (string).
    const auto prompt = j.find("prompt");
    if (prompt == j.end() || !prompt->is_string()) {
        throw BadRequest("\"prompt\" is required and must be a string");
    }
    spec.prompt = prompt->get<std::string>();

    // Optional scalars. Wrong types are reported rather than silently coerced.
    if (auto it = j.find("max_tokens"); it != j.end()) {
        if (!it->is_number_unsigned()) {
            throw BadRequest("\"max_tokens\" must be a non-negative integer");
        }
        spec.max_tokens = it->get<std::uint32_t>();
    }
    if (auto it = j.find("temperature"); it != j.end()) {
        if (!it->is_number()) {
            throw BadRequest("\"temperature\" must be a number");
        }
        spec.sampling.temperature = it->get<float>();
    }
    if (auto it = j.find("top_p"); it != j.end()) {
        if (!it->is_number()) {
            throw BadRequest("\"top_p\" must be a number");
        }
        spec.sampling.top_p = it->get<float>();
    }
    if (auto it = j.find("top_k"); it != j.end()) {
        if (!it->is_number_integer()) {
            throw BadRequest("\"top_k\" must be an integer");
        }
        spec.sampling.top_k = it->get<std::int32_t>();
    }
    if (auto it = j.find("seed"); it != j.end()) {
        if (!it->is_number_unsigned()) {
            throw BadRequest("\"seed\" must be a non-negative integer");
        }
        spec.sampling.seed = it->get<std::uint32_t>();
    }

    // Optional stop: a single string or an array of strings.
    if (auto it = j.find("stop"); it != j.end()) {
        if (it->is_string()) {
            spec.stop.push_back(it->get<std::string>());
        } else if (it->is_array()) {
            for (const auto& elem : *it) {
                if (!elem.is_string()) {
                    throw BadRequest("\"stop\" array must contain only strings");
                }
                spec.stop.push_back(elem.get<std::string>());
            }
        } else {
            throw BadRequest("\"stop\" must be a string or an array of strings");
        }
    }

    return spec;
}

std::string make_generate_response(std::string_view text, FinishReason reason, const Usage& usage) {
    const json j = {
        {"text", text},
        {"finish_reason", to_string(reason)},
        {"usage",
         {
             {"prompt_tokens", usage.prompt_tokens},
             {"completion_tokens", usage.completion_tokens},
             {"total_tokens", usage.total_tokens()},
         }},
    };
    return j.dump();
}

std::string make_error_response(std::string_view message) {
    const json j = {{"error", message}};
    return j.dump();
}

std::string make_stream_delta(std::string_view delta) {
    const json j = {{"text", delta}};
    return j.dump();
}

std::string make_stream_done(FinishReason reason, const Usage& usage) {
    const json j = {
        {"finish_reason", to_string(reason)},
        {"usage",
         {
             {"prompt_tokens", usage.prompt_tokens},
             {"completion_tokens", usage.completion_tokens},
             {"total_tokens", usage.total_tokens()},
         }},
    };
    return j.dump();
}

}  // namespace microvllm
