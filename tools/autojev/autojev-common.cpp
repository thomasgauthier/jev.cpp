#include "autojev-common.h"

#include "llama.h"
#include "mtmd.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <utility>

namespace autojev {

namespace {

constexpr size_t max_options = 255;

std::string metadata_value(const llama_model * model, const char * key) {
    const int32_t length = llama_model_meta_val_str(model, key, nullptr, 0);
    if (length < 0) {
        throw std::runtime_error(std::string("missing model metadata: ") + key);
    }
    std::vector<char> buffer(static_cast<size_t>(length) + 1);
    const int32_t copied = llama_model_meta_val_str(model, key, buffer.data(), buffer.size());
    if (copied < 0 || copied > length) {
        throw std::runtime_error(std::string("failed to read model metadata: ") + key);
    }
    return std::string(buffer.data(), static_cast<size_t>(copied));
}

double metadata_temperature(const llama_model * model) {
    const std::string value = metadata_value(model, "autojev.temperature");
    char * end = nullptr;
    const double result = std::strtod(value.c_str(), &end);
    if (end != value.c_str() + value.size() || !std::isfinite(result) || result <= 0.0) {
        throw std::runtime_error("invalid AutoJev temperature metadata");
    }
    return result;
}

std::string python_json_dump(const json & value) {
    if (value.is_object()) {
        std::string result = "{";
        bool first = true;
        for (auto it = value.begin(); it != value.end(); ++it) {
            if (!first) {
                result += ", ";
            }
            first = false;
            result += json(it.key()).dump();
            result += ": ";
            result += python_json_dump(it.value());
        }
        result += "}";
        return result;
    }
    if (value.is_array()) {
        std::string result = "[";
        for (size_t i = 0; i < value.size(); ++i) {
            if (i != 0) {
                result += ", ";
            }
            result += python_json_dump(value[i]);
        }
        result += "]";
        return result;
    }
    return value.dump();
}

bool is_falsey(const json & value) {
    if (value.is_null()) {
        return true;
    }
    if (value.is_boolean()) {
        return !value.get<bool>();
    }
    if (value.is_number()) {
        return value.get<double>() == 0.0;
    }
    if (value.is_string()) {
        return value.get_ref<const std::string &>().empty();
    }
    if (value.is_array() || value.is_object()) {
        return value.empty();
    }
    return false;
}

std::string describe(const json & value) {
    return value.is_string() ? value.get<std::string>() : python_json_dump(value);
}

} // namespace

uint32_t metadata_u32(const llama_model * model, const char * key) {
    const std::string value = metadata_value(model, key);
    uint32_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc() || parsed.ptr != value.data() + value.size() || result == 0) {
        throw std::runtime_error(std::string("invalid model metadata: ") + key);
    }
    return result;
}

bool has_classifier_metadata(const llama_model * model) {
    const int32_t length = llama_model_meta_val_str(model, "autojev.format_version", nullptr, 0);
    if (length < 0) {
        return false;
    }
    return metadata_value(model, "autojev.format_version") == "1";
}

std::string service_model_name(const llama_model * model) {
    constexpr std::string_view fallback = "autojev-qwen3.8-27b";
    const int32_t length = llama_model_meta_val_str(model, "general.base_model.0.name", nullptr, 0);
    if (length < 0) {
        return std::string(fallback);
    }

    const std::string base_model = metadata_value(model, "general.base_model.0.name");
    std::string slug;
    slug.reserve(base_model.size());
    for (const unsigned char c : base_model) {
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '.') {
            slug.push_back(c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : static_cast<char>(c));
        } else if (!slug.empty() && slug.back() != '-') {
            slug.push_back('-');
        }
    }
    while (!slug.empty() && slug.back() == '-') {
        slug.pop_back();
    }
    return slug.empty() ? std::string(fallback) : "autojev-" + slug;
}


std::vector<std::string> validate_classifier(const llama_model * model, double & temperature) {
    if (metadata_value(model, "autojev.format_version") != "1") {
        throw std::runtime_error("model is not a supported AutoJev decision checkpoint");
    }
    temperature = metadata_temperature(model);
    if (llama_model_n_cls_out(model) != max_options) {
        throw std::runtime_error("AutoJev model must have exactly 255 classifier outputs");
    }
    std::vector<std::string> codes;
    codes.reserve(max_options);
    for (uint32_t i = 0; i < max_options; ++i) {
        const char * label = llama_model_cls_label(model, i);
        if (label == nullptr || label[0] == '\0') {
            throw std::runtime_error("AutoJev model is missing ordered classifier labels");
        }
        codes.emplace_back(label);
    }
    return codes;
}

option_set build_options(const json & question) {
    const std::string type = question.at("type").get<std::string>();
    option_set result;
    if (type == "choice") {
        const auto & criteria = question.at("criteria");
        if (!criteria.is_object()) {
            throw std::runtime_error("choice criteria must be an object");
        }
        for (auto it = criteria.begin(); it != criteria.end(); ++it) {
            result.keys.push_back(it.key());
            result.descriptions.emplace_back(it.value().is_null()
                ? json(it.key())
                : json(it.key() + ": " + describe(it.value())));
        }
    } else if (type == "score") {
        const auto & criteria = question.at("criteria");
        if (!criteria.is_array()) {
            throw std::runtime_error("score criteria must be an array");
        }
        for (size_t i = 0; i < criteria.size(); ++i) {
            result.keys.push_back(std::to_string(i));
            result.descriptions.push_back(criteria[i]);
        }
    } else if (type == "noul") {
        json criteria = question.value("criteria", json::object());
        if (is_falsey(criteria)) {
            criteria = json::object();
        }
        if (!criteria.is_object()) {
            throw std::runtime_error("noul criteria must be an object");
        }
        const json no = criteria.value("false", json());
        const json yes = criteria.value("true", json());
        result.keys = {"false", "true"};
        result.descriptions.push_back(is_falsey(no) ? json("No / false") : no);
        result.descriptions.push_back(is_falsey(yes) ? json("Yes / true") : yes);
    } else {
        throw std::runtime_error("question type must be choice, noul, or score");
    }
    if (result.keys.empty() || result.keys.size() > max_options) {
        throw std::runtime_error("questions must have 1 to 255 options");
    }
    if (type == "score" && result.keys.size() < 2) {
        throw std::runtime_error("score questions require at least two levels");
    }
    return result;
}

std::string build_user_prompt(const json & state, const json & question,
                              const std::vector<std::string> & codes,
                              const option_set & options, size_t n_images) {
    return build_user_prompt(state, question, codes, options, n_images, mtmd_default_marker());
}

std::string build_user_prompt(const json & state, const json & question,
                              const std::vector<std::string> & codes,
                              const option_set & options, size_t n_images,
                              std::string_view image_marker) {
    std::string result;
    for (size_t i = 0; i < n_images; ++i) {
        result.append(image_marker.data(), image_marker.size());
    }
    result += "State:\n" + describe(state);
    const json instructions = question.value("instructions", json());
    result += "\n\nQuestion:\n" + describe(is_falsey(instructions)
        ? json("Choose the best matching option.")
        : instructions);
    result += "\n\nOptions:\n";
    for (size_t i = 0; i < options.keys.size(); ++i) {
        if (i != 0) {
            result += '\n';
        }
        result += codes[i] + ": " + describe(options.descriptions[i]);
    }
    result += "\n\nReturn only the letter code of the best option.";
    return result;
}

std::vector<unsigned char> decode_base64(std::string_view input) {
    static const std::array<int8_t, 256> table = [] {
        std::array<int8_t, 256> values{};
        values.fill(-1);
        constexpr std::string_view alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (size_t i = 0; i < alphabet.size(); ++i) {
            values[static_cast<unsigned char>(alphabet[i])] = static_cast<int8_t>(i);
        }
        return values;
    }();
    if (input.size() % 4 != 0) {
        throw std::runtime_error("image data URI has invalid base64 padding");
    }
    size_t padding = 0;
    if (!input.empty() && input.back() == '=') {
        padding = 1;
        if (input.size() > 1 && input[input.size() - 2] == '=') {
            padding = 2;
        }
    }
    const size_t useful = input.size() - padding;
    if ((padding == 1 && useful % 4 != 3) || (padding == 2 && useful % 4 != 2)) {
        throw std::runtime_error("image data URI has invalid base64 padding");
    }

    std::vector<unsigned char> output;
    output.reserve(input.size() / 4 * 3 - padding);
    uint32_t accumulator = 0;
    int bits = 0;
    for (size_t i = 0; i < useful; ++i) {
        const auto value = table[static_cast<unsigned char>(input[i])];
        if (value < 0) {
            throw std::runtime_error("image data URI contains invalid base64");
        }
        accumulator = (accumulator << 6) | static_cast<uint32_t>(value);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            output.push_back(static_cast<unsigned char>((accumulator >> bits) & 0xff));
        }
    }
    if (output.size() != input.size() / 4 * 3 - padding) {
        throw std::runtime_error("image data URI has invalid base64 content");
    }
    return output;
}

std::vector<double> probabilities(const std::vector<float> & logits, size_t count, double temperature) {
    if (count == 0 || count > logits.size()) {
        throw std::runtime_error("classifier output count does not match question options");
    }
    double maximum = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(logits[i])) {
            throw std::runtime_error("classifier returned a non-finite score");
        }
        maximum = std::max(maximum, static_cast<double>(logits[i]));
    }
    std::vector<double> result(count);
    double total = 0.0;
    for (size_t i = 0; i < count; ++i) {
        result[i] = std::exp((static_cast<double>(logits[i]) - maximum) / temperature);
        total += result[i];
    }
    if (!std::isfinite(total) || total <= 0.0) {
        throw std::runtime_error("classifier probability mass is invalid");
    }
    for (double & value : result) {
        value /= total;
    }
    return result;
}

json make_answer(const json & question, const option_set & options, const std::vector<double> & probs) {
    const std::string type = question.at("type").get<std::string>();
    json result;
    result["type"] = type;
    if (type == "noul") {
        result["noul"] = probs[1];
        return result;
    }

    size_t best = 0;
    for (size_t i = 1; i < probs.size(); ++i) {
        if (probs[i] > probs[best]) {
            best = i;
        }
    }
    json distribution = json::object();
    for (size_t i = 0; i < probs.size(); ++i) {
        distribution[options.keys[i]] = probs[i];
    }
    result["probabilities"] = std::move(distribution);

    if (type == "choice") {
        result["choice"] = options.keys[best];
        const double confidence = probs.size() == 1
            ? 1.0
            : (probs[best] - 1.0 / probs.size()) / (1.0 - 1.0 / probs.size());
        result["confidence"] = std::max(0.0, std::min(1.0, confidence));
        return result;
    }

    json legend = json::object();
    double score = 0.0;
    double distance = 0.0;
    const double midpoint = (probs.size() - 1) / 2.0;
    double baseline = 0.0;
    for (size_t i = 0; i < probs.size(); ++i) {
        const std::string key = std::to_string(i);
        legend[key] = options.descriptions[i];
        score += i * probs[i];
        distance += probs[i] * std::abs(static_cast<double>(i) - best);
        baseline += std::abs(static_cast<double>(i) - midpoint) / probs.size();
    }
    result["legend"] = std::move(legend);
    result["score"] = score;
    result["confidence"] = std::max(0.0, 1.0 - distance / baseline);
    return result;
}

} // namespace autojev
