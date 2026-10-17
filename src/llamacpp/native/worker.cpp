#include "base64.hpp"
#include "chat.h"
#include "ggml-backend.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using json = common_json;

// keep diagnostics off the JSON channel and routine logging out of timing
static void log_errors(ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        std::cerr << text;
    }
}

class decision_worker {
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model{nullptr, llama_model_free};
    std::unique_ptr<llama_context, decltype(&llama_free)> context{nullptr, llama_free};
    std::unique_ptr<mtmd_context, decltype(&mtmd_free)> vision{nullptr, mtmd_free};
    std::unique_ptr<common_chat_template> chat;
    std::vector<llama_token> letters;
    json temperatures = json::object();
    std::string device;

    std::string metadata(const std::string & key) const {
        const int size = llama_model_meta_val_str(model.get(), key.c_str(), nullptr, 0);
        if (size < 0) {
            throw std::runtime_error("missing GGUF metadata: " + key);
        }
        std::vector<char> buffer(size + 1);
        llama_model_meta_val_str(model.get(), key.c_str(), buffer.data(), buffer.size());
        return std::string(buffer.data(), size);
    }

    std::vector<llama_token> tokenize(const std::string & text, bool special) const {
        const auto * vocab = llama_model_get_vocab(model.get());
        int count = llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, special, true);
        if (count >= 0) {
            throw std::runtime_error("empty tokenization result");
        }
        std::vector<llama_token> tokens(-count);
        count = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), special, true);
        if (count < 0) {
            throw std::runtime_error("tokenization failed");
        }
        tokens.resize(count);
        return tokens;
    }

public:
    decision_worker(const char * path, const char * projector, int ctx_size, int gpu_layers, int threads) {
        llama_log_set(log_errors, nullptr);
        mtmd_helper_log_set(log_errors, nullptr);
        ggml_backend_load_all();
        llama_backend_init();
        auto params = llama_model_default_params();
        params.n_gpu_layers = gpu_layers;
        ggml_backend_dev_t cpu_only[] = {nullptr};
        if (gpu_layers == 0) {
            params.devices = cpu_only;
        }
        model.reset(llama_model_load_from_file(path, params));
        if (!model) {
            throw std::runtime_error("could not load GGUF model");
        }
        auto ctx_params = llama_context_default_params();
        ctx_params.n_ctx = ctx_size;
        ctx_params.n_seq_max = 1;
        ctx_params.n_batch = std::min(ctx_size, 2048);
        ctx_params.n_ubatch = std::min(ctx_size, 512);
        if (threads > 0) {
            ctx_params.n_threads = threads;
            ctx_params.n_threads_batch = threads;
        }
        if (gpu_layers == 0) {
            ctx_params.offload_kqv = false;
            ctx_params.op_offload = false;
        }
        context.reset(llama_init_from_model(model.get(), ctx_params));
        if (!context) {
            throw std::runtime_error("could not create model context");
        }
        if (projector[0]) {
            auto image_params = mtmd_context_params_default();
            image_params.use_gpu = gpu_layers != 0;
            image_params.print_timings = false;
            image_params.warmup = false;
            image_params.n_threads = llama_n_threads_batch(context.get());
            vision.reset(mtmd_init_from_file(projector, model.get(), image_params));
            if (!vision) {
                throw std::runtime_error("could not load vision projector");
            }
        }
        const auto arch = metadata("general.architecture");
        if (metadata(arch + ".decision.type") != "openjev") {
            throw std::runtime_error("expected an exported openjev decision model");
        }
        for (const std::string kind : {"choice", "noul", "score"}) {
            const float temperature = std::stof(metadata(arch + ".decision.temperature." + kind));
            if (!std::isfinite(temperature) || temperature <= 0) {
                throw std::runtime_error("invalid saved temperature");
            }
            temperatures[kind] = temperature;
        }
        const char * source = llama_model_chat_template(model.get(), "systemone");
        if (!source) {
            throw std::runtime_error("GGUF has no systemone template");
        }
        chat = std::make_unique<common_chat_template>(source, "", "");
        for (char letter = 'A'; letter <= 'Z'; ++letter) {
            const auto ids = tokenize(std::string(1, letter), false);
            if (ids.size() != 1) {
                throw std::runtime_error("option letter is not a single token");
            }
            letters.push_back(ids[0]);
        }
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto dev = ggml_backend_dev_get(i);
            const auto type = ggml_backend_dev_type(dev);
            if (type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                if (!device.empty()) { device += " + "; }
                device += std::string("CPU (") + ggml_backend_dev_description(dev) + ")";
            } else if (gpu_layers != 0 && (type == GGML_BACKEND_DEVICE_TYPE_GPU ||
                                          type == GGML_BACKEND_DEVICE_TYPE_IGPU)) {
                if (!device.empty()) { device += " + "; }
                device += std::string(ggml_backend_dev_name(dev)) + " (" +
                          ggml_backend_dev_description(dev) + ")";
            }
        }
    }

    json info() const {
        return json{{"temperatures", temperatures}, {"device", device}};
    }

    json predict(json input) {
        const size_t count = input.at("options").size();
        if (count < 2 || count > letters.size()) {
            throw std::runtime_error("unsupported option count");
        }
        const auto files = input.at("files");
        input.erase("files");
        input["images"] = json::array();
        std::vector<std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)>> images;
        std::vector<const mtmd_bitmap *> bitmaps;
        for (const auto & file : files) {
            if (!vision) {
                throw std::runtime_error("image supplied without a vision projector");
            }
            const auto bytes = base64::decode(file.get<std::string>());
            auto bitmap = mtmd_helper_bitmap_init_from_buf(
                vision.get(), reinterpret_cast<const unsigned char *>(bytes.data()), bytes.size(),
                false, mtmd_helper_init_opt_default());
            if (!bitmap.bitmap) {
                throw std::runtime_error("could not decode image");
            }
            images.emplace_back(bitmap.bitmap, mtmd_bitmap_free);
            bitmaps.push_back(bitmap.bitmap);
            input["images"].push_back(mtmd_default_marker());
        }
        jinja::context template_context(chat->source());
        jinja::global_from_json(template_context, input, false);
        jinja::runtime runtime(template_context);
        const auto prompt = jinja::runtime::gather_string_parts(runtime.execute(chat->prog))->as_string().str();

        std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)> chunks(
            nullptr, mtmd_input_chunks_free);
        std::vector<llama_token> tokens;
        size_t token_count;
        llama_pos positions;
        if (vision) {
            chunks.reset(mtmd_input_chunks_init());
            mtmd_input_text text{prompt.data(), prompt.size(), true, true};
            if (mtmd_tokenize(vision.get(), chunks.get(), &text, bitmaps.data(), bitmaps.size()) != 0) {
                throw std::runtime_error("multimodal tokenization failed");
            }
            token_count = mtmd_helper_get_n_tokens(chunks.get());
            positions = mtmd_helper_get_n_pos(chunks.get());
        } else {
            tokens = tokenize(prompt, true);
            token_count = tokens.size();
            positions = tokens.size();
        }
        if (token_count == 0 || token_count > llama_n_ctx(context.get()) ||
            positions > static_cast<llama_pos>(llama_n_ctx(context.get()))) {
            throw std::runtime_error("question exceeds ctx_size; increase llamacpp.ctx_size");
        }
        // preparation and memory reset are outside the inference clock
        llama_memory_clear(llama_get_memory(context.get()), true);
        llama_synchronize(context.get());
        const auto start = std::chrono::steady_clock::now();
        int status = 0;
        if (vision) {
            llama_pos new_past = 0;
            status = mtmd_helper_eval_chunks(vision.get(), context.get(), chunks.get(), 0, 0,
                                             llama_n_batch(context.get()), true, &new_past);
        } else {
            for (size_t offset = 0; offset < tokens.size() && status == 0;) {
                const auto size = std::min(tokens.size() - offset, size_t(llama_n_batch(context.get())));
                auto batch = llama_batch_get_one(tokens.data() + offset, size);
                status = llama_decode(context.get(), batch);
                offset += size;
            }
        }
        llama_synchronize(context.get());
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (status != 0) {
            throw std::runtime_error("native forward pass failed: " + std::to_string(status));
        }
        // scoring, JSON serialization, and transport are outside the clock
        const float * raw = llama_get_logits_ith(context.get(), -1);
        if (!raw) {
            throw std::runtime_error("model returned no final-token logits");
        }
        json logits = json::array();
        for (size_t i = 0; i < count; ++i) {
            if (!std::isfinite(raw[letters[i]])) {
                throw std::runtime_error("model returned non-finite logits");
            }
            logits.push_back(raw[letters[i]]);
        }
        return json{{"logits", logits}, {"seconds", seconds}, {"input_tokens", token_count}};
    }
};

int main(int argc, char ** argv) {
    try {
        if (argc != 6) {
            throw std::runtime_error("expected model, projector, context size, GPU layers, threads");
        }
        decision_worker worker(argv[1], argv[2], std::stoi(argv[3]), std::stoi(argv[4]), std::stoi(argv[5]));
        std::cout << worker.info().dump() << std::endl;
        std::string line;
        while (std::getline(std::cin, line)) {
            std::cout << worker.predict(json::parse(line)).dump() << std::endl;
        }
        return 0;
    } catch (const std::exception & error) {
        std::cout << json{{"error", error.what()}}.dump() << std::endl;
        return 1;
    }
}
