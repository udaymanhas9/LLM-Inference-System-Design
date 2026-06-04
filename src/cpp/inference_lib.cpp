#include "inference_lib.h"
#include "llama.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

// ── Internal context ─────────────────────────────────────────────────────────

struct InferenceCtx {
    llama_model*   model;
    llama_context* ctx;
    int            n_ctx;
};

// ── Lifecycle ─────────────────────────────────────────────────────────────────

InferenceCtx* inference_create(const char* model_path, int n_gpu_layers, int n_ctx) {
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;  // -1 = all layers on Metal (M4)

    llama_model* model = llama_model_load_from_file(model_path, mparams);
    if (!model) return nullptr;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = static_cast<uint32_t>(n_ctx);
    cparams.n_batch = 512;   // max tokens per llama_decode call

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_model_free(model);
        return nullptr;
    }

    return new InferenceCtx{model, ctx, n_ctx};
}

void inference_destroy(InferenceCtx* h) {
    if (!h) return;
    llama_free(h->ctx);
    llama_model_free(h->model);
    delete h;
    llama_backend_free();
}

// ── Generation ───────────────────────────────────────────────────────────────

int inference_generate(
    InferenceCtx* h,
    const char*   prompt,
    int           max_tokens,
    int           prefill_chunk_size,
    float         temperature,
    token_cb_t    callback,
    void*         userdata
) {
    if (!h || !prompt || !callback) return -1;

    // ── Tokenise ──────────────────────────────────────────────────────────────
    const int n_max = h->n_ctx - max_tokens - 4;
    std::vector<llama_token> tokens(n_max);
    int n_tokens = llama_tokenize(
        llama_model_get_vocab(h->model),
        prompt, static_cast<int32_t>(strlen(prompt)),
        tokens.data(), static_cast<int32_t>(tokens.size()),
        /*add_special=*/true,
        /*parse_special=*/true
    );
    if (n_tokens < 0) return -2;
    tokens.resize(n_tokens);

    // ── Clear KV cache ────────────────────────────────────────────────────────
    llama_memory_clear(llama_get_memory(h->ctx), /*data=*/true);

    // ── Chunked prefill ───────────────────────────────────────────────────────
    //
    // Instead of one large llama_decode(all_tokens), we split the prompt into
    // slices of `prefill_chunk_size` tokens.  Each slice is one forward pass.
    //
    // Why: in a multi-sequence system you interleave one decode step between
    // each prefill chunk so in-flight sequences don't stall for the full prefill
    // duration.  Here we have one sequence, so the chunks just limit peak memory
    // pressure on the KV cache during prefill — but the mechanism is identical.
    //
    const int chunk = (prefill_chunk_size > 0) ? prefill_chunk_size : n_tokens;
    for (int offset = 0; offset < n_tokens; offset += chunk) {
        const int n = std::min(chunk, n_tokens - offset);
        llama_batch batch = llama_batch_get_one(tokens.data() + offset, n);
        if (llama_decode(h->ctx, batch) != 0) return -3;
    }

    // ── Sampler chain ─────────────────────────────────────────────────────────
    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (temperature > 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    }

    // ── Autoregressive decode loop ────────────────────────────────────────────
    //
    // This is the decode phase of continuous batching: each iteration produces
    // one token.  In a real multi-sequence system `batch` would contain one
    // decode token per in-flight sequence — each assigned its own seq_id.
    // Here we have a single sequence (seq_id=0) for simplicity.
    //
    int n_pos = n_tokens;
    char piece_buf[256];

    for (int i = 0; i < max_tokens; i++) {
        // Sample
        llama_token tok = llama_sampler_sample(smpl, h->ctx, -1);
        llama_sampler_accept(smpl, tok);

        // End-of-generation?
        if (llama_vocab_is_eog(llama_model_get_vocab(h->model), tok)) break;

        // Detokenise to UTF-8 bytes and fire callback
        int len = llama_token_to_piece(
            llama_model_get_vocab(h->model), tok,
            piece_buf, static_cast<int32_t>(sizeof(piece_buf) - 1),
            /*lstrip=*/0,
            /*special=*/false
        );
        if (len > 0) {
            piece_buf[len] = '\0';
            callback(piece_buf, 0, userdata);
        }

        // Feed token back for next step
        llama_batch next = llama_batch_get_one(&tok, 1);
        if (llama_decode(h->ctx, next) != 0) break;
        n_pos++;
    }

    callback("", /*is_done=*/1, userdata);
    llama_sampler_free(smpl);
    return 0;
}
