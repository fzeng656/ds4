#import <Foundation/Foundation.h>
#include "../qn_qwen4_model.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QN_WORKER_MAX_PROMPT 8192u
#define QN_WORKER_MAX_OUTPUT 2048u
#define QN_WORKER_MAX_STOPS 32u

static int parse_u32(const char *s, uint32_t *out) {
    if (!s || !*s) return -1;
    errno = 0; char *end = NULL; unsigned long v = strtoul(s, &end, 10);
    if (errno || !end || *end || v > UINT32_MAX) return -1;
    *out = (uint32_t)v; return 0;
}
static int is_stop(uint32_t id, const uint32_t *stops, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) if (stops[i] == id) return 1;
    return 0;
}
static char *next_tok(char **save) { return strtok_r(NULL, " \t\r\n", save); }

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s MODEL_DIR MANIFEST NGRAM_TABLE\n", argv[0]);
        return 64;
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    setenv("QN_RUNTIME_MODE", "stable", 1);
    @autoreleasepool {
        char err[1024] = {0};
        qn_qwen4_model_config cfg = {argv[1], argv[2], argv[3], 16};
        qn_qwen4_model *model = NULL;
        if (qn_qwen4_model_open(&model, &cfg, err, sizeof(err))) {
            fprintf(stderr, "worker open: %s\n", err); return 1;
        }
        if (qn_qwen4_model_prepare_production(model, err, sizeof(err))) {
            fprintf(stderr, "worker prepare: %s\n", err); qn_qwen4_model_close(model); return 2;
        }
        qn_qwen4_runtime_status rs = {0}; qn_qwen4_model_get_runtime_status(model, &rs);
        printf("READY stable mps=%u+%u bm32=%u+%u\n",
               rs.gdn_mps_layers, rs.qsa_mps_layers, rs.gdn_bm32_layers, rs.qsa_bm32_layers);

        uint32_t *cached_tokens = calloc(QN_WORKER_MAX_PROMPT, sizeof(uint32_t));
        uint32_t cached_n = 0, cached_next = 0; int cached_next_valid = 0;
        if (!cached_tokens) { qn_qwen4_model_close(model); return 70; }
        char *line = NULL; size_t cap = 0;
        while (getline(&line, &cap, stdin) >= 0) {
            char *save = NULL; char *cmd = strtok_r(line, " \t\r\n", &save);
            if (!cmd) continue;
            if (!strcmp(cmd, "PING")) { printf("PONG\n"); continue; }
            if (!strcmp(cmd, "RESET")) { if (qn_qwen4_model_reset_session(model, err, sizeof(err))) { printf("FATAL reset_failed %s\n", err); free(cached_tokens); free(line); qn_qwen4_model_close(model); return 3; } cached_n=0; cached_next_valid=0; printf("RESET\n"); continue; }
            if (!strcmp(cmd, "QUIT")) { printf("BYE\n"); break; }
            if (strcmp(cmd, "GEN")) { printf("ERR 400 unknown_command\n"); continue; }

            uint32_t max_tokens = 0, nstops = 0, nprompt = 0;
            char *x = next_tok(&save); if (parse_u32(x, &max_tokens) || !max_tokens || max_tokens > QN_WORKER_MAX_OUTPUT) { printf("ERR 400 invalid_max_tokens\n"); continue; }
            x = next_tok(&save); if (parse_u32(x, &nstops) || nstops > QN_WORKER_MAX_STOPS) { printf("ERR 400 invalid_stop_count\n"); continue; }
            uint32_t stops[QN_WORKER_MAX_STOPS]; int bad = 0;
            for (uint32_t i = 0; i < nstops; ++i) { x = next_tok(&save); if (parse_u32(x, &stops[i])) { bad = 1; break; } }
            if (bad) { printf("ERR 400 invalid_stop_id\n"); continue; }
            x = next_tok(&save); if (parse_u32(x, &nprompt) || !nprompt || nprompt > QN_WORKER_MAX_PROMPT) { printf("ERR 400 invalid_prompt_count\n"); continue; }
            if ((uint64_t)nprompt + (uint64_t)max_tokens > 8192u) { printf("ERR 400 context_limit_exceeded\n"); continue; }
            uint32_t *prompt = malloc((size_t)nprompt * sizeof(uint32_t));
            if (!prompt) { printf("FATAL allocation_failed\n"); free(line); qn_qwen4_model_close(model); return 70; }
            for (uint32_t i = 0; i < nprompt; ++i) { x = next_tok(&save); if (parse_u32(x, &prompt[i])) { bad = 1; break; } }
            if (!bad && next_tok(&save)) bad = 1;
            if (bad) { free(prompt); printf("ERR 400 invalid_prompt_ids\n"); continue; }

            uint32_t common = 0; while (common < cached_n && common < nprompt && cached_tokens[common] == prompt[common]) common++;
            uint32_t reused = (common == cached_n) ? cached_n : 0;
            qn_qwen4_prefill_output pf = {0}; double prefill_ms = 0.0;
            if (!reused) {
                if (qn_qwen4_model_reset_session(model, err, sizeof(err))) {
                    free(prompt); printf("FATAL reset_failed %s\n", err); free(cached_tokens); free(line); qn_qwen4_model_close(model); return 3;
                }
                cached_n=0; cached_next_valid=0;
            }
            if (reused < nprompt) {
                if (qn_qwen4_model_prefill_tokens(model, prompt + reused, nprompt - reused, NULL, &pf, err, sizeof(err))) {
                    free(prompt); printf("FATAL prefill_failed %s\n", err); free(cached_tokens); free(line); qn_qwen4_model_close(model); return 4;
                }
                prefill_ms = pf.prefill_ms; cached_next = pf.next_token; cached_next_valid = 1;
                memcpy(cached_tokens + reused, prompt + reused, (size_t)(nprompt-reused)*sizeof(uint32_t)); cached_n=nprompt;
            } else if (!cached_next_valid) {
                free(prompt); printf("FATAL cache_prediction_missing\n"); free(cached_tokens); free(line); qn_qwen4_model_close(model); return 4;
            }
            free(prompt);
            printf("BEGIN %u %.3f %u\n", nprompt, prefill_ms, reused);
            uint32_t tok = cached_next; double decode_ms = 0.0; uint32_t produced = 0; int stopped = 0;
            for (;;) {
                printf("TOK %u %.3f\n", tok, produced ? decode_ms : 0.0);
                produced++;
                /* Finalize every emitted token into the persistent model state so
                   cached_tokens exactly describes the state carried to the next request. */
                qn_qwen4_step_output so = {0};
                if (qn_qwen4_model_step(model, tok, NULL, &so, err, sizeof(err))) {
                    printf("FATAL decode_failed %s\n", err); free(cached_tokens); free(line); qn_qwen4_model_close(model); return 5;
                }
                decode_ms += so.total_ms;
                if (cached_n >= QN_WORKER_MAX_PROMPT) { printf("FATAL cache_context_overflow\n"); free(cached_tokens); free(line); qn_qwen4_model_close(model); return 6; }
                cached_tokens[cached_n++] = tok; cached_next = so.next_token; cached_next_valid = 1;
                if (is_stop(tok, stops, nstops)) { stopped = 1; break; }
                if (produced >= max_tokens) break;
                tok = cached_next;
            }
            printf("END %s %u %.3f\n", stopped ? "stop" : "length", produced, decode_ms);
        }
        free(cached_tokens); free(line); qn_qwen4_model_close(model); return 0;
    }
}
