#import <Foundation/Foundation.h>
#include "../qn_qwen4_model.h"
#include <stdio.h>

int main(void) {
    @autoreleasepool {
        const char *model_dir = "/Users/mikuru/qwen38fn-official-pipeline/output/Qwen3.8-Flash-Next-OurAblit-Mixed-MLX-Serve";
        const char *manifest = "/Users/mikuru/qwen-native-runtime/phase0/qwen38fn_manifest.json";
        char ngram[1400], err[1024] = {0};
        snprintf(ngram, sizeof(ngram), "%s/ngram_table.bin", model_dir);
        qn_qwen4_model_config cfg = {model_dir, manifest, ngram, 16};
        qn_qwen4_model *model = NULL;
        if (qn_qwen4_model_open(&model, &cfg, err, sizeof(err))) {
            fprintf(stderr, "open: %s\n", err); return 1;
        }

        /* "The capital of France is" with add_special_tokens=false. */
        const uint32_t prompt[] = {760, 6511, 314, 9338, 369};
        const uint32_t expected[] = {11751, 13, 561, 6511, 314};
        qn_qwen4_prefill_output pf = {0};
        if (qn_qwen4_model_prefill_tokens(model, prompt, 5, NULL, &pf, err, sizeof(err))) {
            fprintf(stderr, "prefill: %s\n", err); qn_qwen4_model_close(model); return 2;
        }
        int mismatch = pf.next_token != expected[0];
        uint32_t token = pf.next_token;
        printf("prefill tokens=%u next=%u expected=%u prefill=%.3fms logits=%.3fms\n",
               pf.prompt_tokens, token, expected[0], pf.prefill_ms, pf.logits_ms);
        for (int i = 1; i < 5; ++i) {
            qn_qwen4_step_output out = {0};
            if (qn_qwen4_model_step(model, token, NULL, &out, err, sizeof(err))) {
                fprintf(stderr, "decode %d: %s\n", i, err); qn_qwen4_model_close(model); return 3;
            }
            token = out.next_token;
            printf("decode%d next=%u expected=%u total=%.3fms\n", i, token, expected[i], out.total_ms);
            mismatch += token != expected[i];
        }
        qn_qwen4_model_stats stats = {0};
        qn_qwen4_model_get_stats(model, &stats);
        printf("mismatch=%d position=%u qsa_cache=%.2fMiB\n", mismatch, stats.position,
               (double)stats.qsa_cache_bytes / 1048576.0);
        qn_qwen4_model_close(model);
        return mismatch ? 4 : 0;
    }
}
