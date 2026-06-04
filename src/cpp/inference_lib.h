#pragma once
#ifdef __cplusplus
extern "C" {
#endif

typedef struct InferenceCtx InferenceCtx;

/*
 * Fires for each generated token.  text is a null-terminated UTF-8 string.
 * is_done == 1 on the final call (text is empty); use it as a sentinel.
 */
typedef void (*token_cb_t)(const char* text, int is_done, void* userdata);

/*
 * Load a GGUF model file.
 *   n_gpu_layers : number of transformer layers to offload to Metal/CUDA.
 *                  Pass -1 to offload ALL layers (recommended on M4 with enough RAM).
 *   n_ctx        : KV-cache context window size in tokens.
 */
InferenceCtx* inference_create(const char* model_path, int n_gpu_layers, int n_ctx);
void          inference_destroy(InferenceCtx* ctx);

/*
 * Generate tokens for `prompt`.
 *
 *   prefill_chunk_size : max tokens processed per llama_decode call during prefill.
 *                        This is chunked prefill: prevents one long prefill from
 *                        monopolising the forward pass.  0 = process all at once.
 *   temperature        : sampling temperature.  0.0 = greedy.
 *
 * Calls `callback` once per token, then once with is_done=1.
 * Returns 0 on success, negative on error.
 */
int inference_generate(
    InferenceCtx* ctx,
    const char*   prompt,
    int           max_tokens,
    int           prefill_chunk_size,
    float         temperature,
    token_cb_t    callback,
    void*         userdata
);

#ifdef __cplusplus
}
#endif
