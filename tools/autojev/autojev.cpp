#include "autojev-common.h"
#include "chat.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using autojev::build_options;
using autojev::build_user_prompt;
using autojev::decode_base64;
using autojev::json;
using autojev::make_answer;
using autojev::metadata_u32;
using autojev::option_set;
using autojev::probabilities;
using autojev::system_prompt;
using autojev::validate_classifier;

struct arguments {
    std::string model_path;
    std::string mmproj_path;
    int32_t gpu_layers = -1;
    uint32_t context_size = 8192;
    uint32_t batch_size = 2048;
    bool trace_prompts = false;
};

void print_usage(const char * program) {
    std::cerr << "Usage: " << program << " --model MODEL.gguf --mmproj MMPROJ.gguf [options]\n"
              << "Read AutoJev JSONL requests on stdin; write one JSON response per line.\n"
              << "  --gpu-layers N   layers to offload (-1: all; default -1)\n"
              << "  --ctx-size N     maximum context tokens (default 8192)\n"
              << "  --batch-size N   decode batch size (default 2048)\n"
              << "  --trace-prompts  write rendered prompts and token IDs to stderr\n";
}

uint32_t parse_u32(const std::string & value, const char * option) {
    uint32_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc() || parsed.ptr != value.data() + value.size() || result == 0) {
        throw std::runtime_error(std::string("invalid value for ") + option + ": " + value);
    }
    return result;
}

int32_t parse_i32(const std::string & value, const char * option) {
    int32_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc() || parsed.ptr != value.data() + value.size() || result < -1) {
        throw std::runtime_error(std::string("invalid value for ") + option + ": " + value);
    }
    return result;
}

arguments parse_arguments(int argc, char ** argv) {
    arguments result;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        auto value = [&](const char * name) -> std::string {
            if (++i >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[i];
        };
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--model" || arg == "-m") {
            result.model_path = value("--model");
        } else if (arg == "--mmproj") {
            result.mmproj_path = value("--mmproj");
        } else if (arg == "--gpu-layers") {
            result.gpu_layers = parse_i32(value("--gpu-layers"), "--gpu-layers");
        } else if (arg == "--ctx-size") {
            result.context_size = parse_u32(value("--ctx-size"), "--ctx-size");
        } else if (arg == "--batch-size") {
            result.batch_size = parse_u32(value("--batch-size"), "--batch-size");
        } else if (arg == "--trace-prompts") {
            result.trace_prompts = true;
        } else {
            throw std::runtime_error("unknown option: " + std::string(arg));
        }
    }
    if (result.model_path.empty() || result.mmproj_path.empty()) {
        throw std::runtime_error("--model and --mmproj are required");
    }
    return result;
}

std::vector<unsigned char> image_bytes(const std::string & source) {
    if (source.rfind("data:image/", 0) == 0) {
        const size_t comma = source.find(',');
        if (comma == std::string::npos) {
            throw std::runtime_error("image data URI has no payload separator");
        }
        return decode_base64(std::string_view(source).substr(comma + 1));
    }
    std::ifstream input(source, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open image: " + source);
    }
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (bytes.empty()) {
        throw std::runtime_error("image is empty: " + source);
    }
    return bytes;
}

std::vector<mtmd::bitmap_ptr> load_images(const mtmd_context * vision, const json & values) {
    if (!values.is_array()) {
        throw std::runtime_error("images must be an array");
    }
    std::vector<mtmd::bitmap_ptr> result;
    result.reserve(values.size());
    for (const auto & value : values) {
        if (!value.is_string()) {
            throw std::runtime_error("each image must be a data URI or a file path");
        }
        const std::string source = value.get<std::string>();
        const auto bytes = image_bytes(source);
        const auto wrapper = mtmd_helper_bitmap_init_from_buf(
            vision, bytes.data(), bytes.size(), false, mtmd_helper_init_opt_default());
        if (wrapper.bitmap == nullptr) {
            throw std::runtime_error("failed to decode image input");
        }
        if (wrapper.video_ctx != nullptr) {
            mtmd_helper_video_free(wrapper.video_ctx);
        }
        result.emplace_back(wrapper.bitmap);
    }
    return result;
}

std::vector<float> evaluate_question(mtmd_context * vision, llama_context * context,
                                     const llama_model * model,
                                     const std::vector<mtmd::bitmap_ptr> & images,
                                     const std::string & prompt,
                                     int32_t batch_size,
                                     bool trace,
                                     const std::string & question_id,
                                     size_t & token_count) {
    mtmd::input_chunks_ptr chunks(mtmd_input_chunks_init());
    if (!chunks) {
        throw std::runtime_error("failed to allocate multimodal input chunks");
    }
    const mtmd_input_text input_text = {prompt.data(), prompt.size(), false, true};
    std::vector<const mtmd_bitmap *> bitmaps;
    bitmaps.reserve(images.size());
    for (const auto & image : images) {
        bitmaps.push_back(image.get());
    }
    const int32_t tokenize_result = mtmd_tokenize(
        vision, chunks.get(), &input_text, bitmaps.data(), bitmaps.size());
    if (tokenize_result != 0) {
        throw std::runtime_error("failed to tokenize AutoJev prompt (MTMD error " + std::to_string(tokenize_result) + ")");
    }

    token_count = mtmd_helper_get_n_tokens(chunks.get());
    if (token_count == 0) {
        throw std::runtime_error("AutoJev prompt produced no model tokens");
    }
    if (trace) {
        json trace_data;
        trace_data["question_id"] = question_id;
        trace_data["prompt"] = prompt;
        trace_data["text_token_ids"] = json::array();
        trace_data["media_token_counts"] = json::array();
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks.get()); ++i) {
            const auto * chunk = mtmd_input_chunks_get(chunks.get(), i);
            if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                size_t n_tokens = 0;
                const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
                for (size_t j = 0; j < n_tokens; ++j) {
                    trace_data["text_token_ids"].push_back(tokens[j]);
                }
            } else {
                trace_data["media_token_counts"].push_back(mtmd_input_chunk_get_n_tokens(chunk));
            }
        }
        trace_data["total_tokens"] = token_count;
        std::cerr << "AUTOJEV_TRACE " << trace_data.dump() << '\n';
    }

    llama_memory_clear(llama_get_memory(context), true);
    llama_pos new_n_past = 0;
    const int32_t eval_result = mtmd_helper_eval_chunks(
        vision, context, chunks.get(), 0, 0, batch_size, true, &new_n_past);
    if (eval_result != 0) {
        throw std::runtime_error("failed to evaluate AutoJev prompt (MTMD error " + std::to_string(eval_result) + ")");
    }
    if (new_n_past <= 0) {
        throw std::runtime_error("AutoJev prompt evaluation produced no context positions");
    }

    const float * output = llama_get_embeddings_seq(context, 0);
    if (output == nullptr) {
        throw std::runtime_error("model returned no classifier scores");
    }
    const uint32_t n_cls_out = llama_model_n_cls_out(model);
    std::vector<float> logits(output, output + n_cls_out);
    if (trace) {
        json trace_data;
        trace_data["question_id"] = question_id;
        trace_data["logits"] = logits;
        std::cerr << "AUTOJEV_SCORES " << trace_data.dump() << '\n';
    }
    return logits;
}

int run(const arguments & args) {
    llama_backend_init();
    struct backend_guard {
        ~backend_guard() { llama_backend_free(); }
    } backend;

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = args.gpu_layers;
    model_params.main_gpu = 0;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(args.model_path.c_str(), model_params), llama_model_free);
    if (!model) {
        throw std::runtime_error("failed to load classifier model: " + args.model_path);
    }

    double temperature = 0.0;
    std::vector<std::string> codes = validate_classifier(model.get(), temperature);

    mtmd_context_params vision_params = mtmd_context_params_default();
    vision_params.use_gpu = true;
    vision_params.image_min_tokens = static_cast<int>(metadata_u32(model.get(), "autojev.image_min_tokens"));
    vision_params.image_max_tokens = static_cast<int>(metadata_u32(model.get(), "autojev.image_max_tokens"));
    std::unique_ptr<mtmd_context, mtmd::mtmd_context_deleter> vision(
        mtmd_init_from_file(args.mmproj_path.c_str(), model.get(), vision_params));
    if (!vision) {
        throw std::runtime_error("failed to load multimodal projector: " + args.mmproj_path);
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = args.context_size;
    context_params.n_batch = args.batch_size;
    context_params.n_ubatch = std::min<uint32_t>(args.batch_size, 512);
    context_params.n_seq_max = 1;
    context_params.type_k = GGML_TYPE_Q8_0;
    context_params.type_v = GGML_TYPE_Q8_0;
    context_params.pooling_type = LLAMA_POOLING_TYPE_RANK;
    context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    context_params.embeddings = true;
    std::unique_ptr<llama_context, decltype(&llama_free)> context(
        llama_init_from_model(model.get(), context_params), llama_free);
    if (!context) {
        throw std::runtime_error("failed to create classifier context");
    }

    const common_chat_templates_ptr templates = common_chat_templates_init(model.get(), "");
    if (!templates) {
        throw std::runtime_error("failed to load model chat template");
    }

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }
        const json request = json::parse(line);
        if (!request.is_object() || !request.contains("state") || !request.at("questions").is_object()) {
            throw std::runtime_error("each JSONL request must contain state and a questions object");
        }
        const json images_value = request.value("images", json::array());
        const auto images = load_images(vision.get(), images_value);
        json answers = json::object();
        size_t request_tokens = 0;
        for (auto it = request.at("questions").begin(); it != request.at("questions").end(); ++it) {
            const std::string question_id = it.key();
            const json & question = it.value();
            const option_set options = build_options(question);
            const std::string user_prompt = build_user_prompt(request.at("state"), question, codes, options, images.size());
            common_chat_msg system;
            system.role = "system";
            system.content = system_prompt;
            common_chat_msg user;
            user.role = "user";
            user.content = user_prompt;
            common_chat_templates_inputs inputs;
            inputs.messages = {system, user};
            inputs.enable_thinking = false;
            inputs.add_generation_prompt = true;
            const common_chat_params rendered = common_chat_templates_apply(templates.get(), inputs);

            size_t question_tokens = 0;
            const auto logits = evaluate_question(vision.get(), context.get(), model.get(), images,
                rendered.prompt, static_cast<int32_t>(args.batch_size), args.trace_prompts,
                question_id, question_tokens);
            request_tokens += question_tokens;
            answers[question_id] = make_answer(question, options, probabilities(logits, options.keys.size(), temperature));
        }
        json response;
        response["answers"] = std::move(answers);
        response["usage"] = json{{"input_tokens", request_tokens}};
        std::cout << response.dump() << '\n';
    }
    if (!std::cin.eof()) {
        throw std::runtime_error("failed while reading JSONL input");
    }
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        return run(parse_arguments(argc, argv));
    } catch (const std::exception & error) {
        std::cerr << "llama-autojev: " << error.what() << '\n';
        return 1;
    }
}
