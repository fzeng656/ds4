#import <Foundation/Foundation.h>
#include "qn_qwen4_model.h"
#include <stdio.h>

int main(void) {
    const char *model_dir = "/Users/mikuru/qwen38fn-official-pipeline/output/Qwen3.8-Flash-Next-OurAblit-Mixed-MLX-Serve";
    const char *manifest = "/Users/mikuru/qwen-native-runtime/phase0/qwen38fn_manifest.json";
    char ngram[1400], err[1024] = {0};
    snprintf(ngram, sizeof(ngram), "%s/ngram_table.bin", model_dir);
    qn_qwen4_model_config cfg = {model_dir, manifest, ngram, 16};
    qn_qwen4_model *model = NULL;
    if (qn_qwen4_model_open(&model, &cfg, err, sizeof(err))) {
        fprintf(stderr, "open: %s\n", err); return 1;
    }
    const uint32_t expected[5] = {10655, 4963, 318, 16, 24};
    uint32_t token = 12345;
    int mismatch = 0;
    for (int step = 0; step < 5; ++step) {
        qn_qwen4_step_output out = {0};
        if (qn_qwen4_model_step(model, token, NULL, &out, err, sizeof(err))) {
            fprintf(stderr, "step %d: %s\n", step, err); qn_qwen4_model_close(model); return 2;
        }
        printf("step%d in=%u out=%u expected=%u trunk=%.3fms logits=%.3fms total=%.3fms\n",
               step, token, out.next_token, expected[step], out.trunk_ms, out.logits_ms, out.total_ms);
        if (out.next_token != expected[step]) mismatch++;
        token = out.next_token;
    }
    qn_qwen4_model_stats stats = {0};
    qn_qwen4_model_get_stats(model, &stats);
    printf("mismatch=%d position=%u qsa_cache=%.2fMiB opened=%u/%u ple=%d\n",
           mismatch, stats.position, (double)stats.qsa_cache_bytes / 1048576.0,
           stats.opened_qsa_layers, stats.opened_gdn_layers, stats.ple_opened);
    qn_qwen4_model_close(model);
    return mismatch ? 3 : 0;
}
