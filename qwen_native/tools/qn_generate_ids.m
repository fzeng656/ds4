#import <Foundation/Foundation.h>
#include "../qn_qwen4_model.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s MODEL_DIR MANIFEST NGRAM MAX_TOKENS TOKEN_ID...\n", argv[0]);
        return 64;
    }
    @autoreleasepool {
        const char *model_dir = argv[1], *manifest = argv[2], *ngram = argv[3];
        int max_tokens = atoi(argv[4]);
        if (max_tokens < 1) return 64;
        size_t n_prompt = (size_t)(argc - 5);
        uint32_t *prompt = calloc(n_prompt, sizeof(uint32_t));
        uint32_t *generated = calloc((size_t)max_tokens, sizeof(uint32_t));
        if (!prompt || !generated) return 70;
        for (size_t i = 0; i < n_prompt; ++i) prompt[i] = (uint32_t)strtoul(argv[5 + i], NULL, 10);

        char err[1024] = {0};
        qn_qwen4_model_config cfg = {model_dir, manifest, ngram, 65536};
        qn_qwen4_model *model = NULL;
        if (qn_qwen4_model_open(&model, &cfg, err, sizeof(err))) {
            fprintf(stderr, "open: %s\n", err); return 1;
        }
        const char *runtime_mode = getenv("QN_RUNTIME_MODE");
        if (runtime_mode && !strcmp(runtime_mode, "stable")) {
            if (qn_qwen4_model_prepare_production(model, err, sizeof(err))) {
                fprintf(stderr, "production prepare: %s\n", err); qn_qwen4_model_close(model); return 5;
            }
            qn_qwen4_runtime_status rs = {0};
            qn_qwen4_model_get_runtime_status(model, &rs);
            fprintf(stderr, "native runtime: stable prepared=%d mps=%u/36+%u/12 bm32=%u/36+%u/12 metallib=%s\n",
                    rs.production_prepared, rs.gdn_mps_layers, rs.qsa_mps_layers,
                    rs.gdn_bm32_layers, rs.qsa_bm32_layers, rs.moe_metallib_path);
        }
        qn_qwen4_prefill_output pf = {0};
        if (qn_qwen4_model_prefill_tokens(model, prompt, n_prompt, NULL, &pf, err, sizeof(err))) {
            fprintf(stderr, "prefill: %s\n", err); qn_qwen4_model_close(model); return 2;
        }
        generated[0] = pf.next_token;
        double decode_ms = 0.0;
        for (int i = 1; i < max_tokens; ++i) {
            qn_qwen4_step_output out = {0};
            if (qn_qwen4_model_step(model, generated[i - 1], NULL, &out, err, sizeof(err))) {
                fprintf(stderr, "decode %d: %s\n", i, err); qn_qwen4_model_close(model); return 3;
            }
            generated[i] = out.next_token;
            decode_ms += out.total_ms;
        }
        printf("TOKENS");
        for (int i = 0; i < max_tokens; ++i) printf(" %u", generated[i]);
        printf("\n");
        fprintf(stderr, "native prefill: %zu tokens %.3f ms; decode: %d tokens %.3f ms\n",
                n_prompt, pf.prefill_ms, max_tokens - 1, decode_ms);
        qn_qwen4_model_close(model);
        free(prompt); free(generated);
        return 0;
    }
}
