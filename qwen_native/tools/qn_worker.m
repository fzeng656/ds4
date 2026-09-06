#import <Foundation/Foundation.h>
#include "../qn_qwen4_model.h"
#include "../qn_mtp_head.h"
#include <mach/mach_time.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QN_WORKER_MAX_PROMPT 32768u
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
static double worker_ms(void) { static mach_timebase_info_data_t tb; if (!tb.denom) mach_timebase_info(&tb); return (double)mach_absolute_time()*tb.numer/tb.denom/1e6; }
static int mtp_env_enabled(void) { const char *v=getenv("QN_MTP"); return v && v[0] && v[0]!='0'; }

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
        int mtp_enabled = mtp_env_enabled();
        qn_model_io *mtp_io = NULL; qn_mtp_head *mtp = NULL;
        if (mtp_enabled) {
            qn_model_io_config ioc = {argv[1], argv[2]};
            if (qn_model_io_open(&mtp_io, &ioc, err, sizeof(err))) { fprintf(stderr, "worker mtp io: %s\n", err); qn_qwen4_model_close(model); return 2; }
            qn_mtp_head_config mhc = {argv[1], argv[2], mtp_io};
            if (qn_mtp_head_open(&mtp, &mhc, err, sizeof(err))) { fprintf(stderr, "worker mtp open: %s\n", err); qn_model_io_close(mtp_io); qn_qwen4_model_close(model); return 2; }
            /* One throw-away step allocates/faults the MTP decode scratch before READY. */
            float wz[10240]={0}, wo[10240]; qn_mtp_head_output wmo={0};
            if (qn_mtp_head_forward(mtp, wz, 0, 0, wo, &wmo, err, sizeof(err)) || qn_mtp_head_reset(mtp)) {
                fprintf(stderr, "worker mtp warm: %s\n", err); qn_mtp_head_close(mtp); qn_model_io_close(mtp_io); qn_qwen4_model_close(model); return 2;
            }
        }
        qn_qwen4_runtime_status rs = {0}; qn_qwen4_model_get_runtime_status(model, &rs);
        printf("READY stable mps=%u+%u bm32=%u+%u mtp=%s\n",
               rs.gdn_mps_layers, rs.qsa_mps_layers, rs.gdn_bm32_layers, rs.qsa_bm32_layers, mtp_enabled?"on":"off");

        uint32_t *cached_tokens = calloc(QN_WORKER_MAX_PROMPT, sizeof(uint32_t));
        uint32_t cached_n = 0, cached_next = 0; int cached_next_valid = 0;
        if (!cached_tokens) { qn_qwen4_model_close(model); return 70; }
        char *line = NULL; size_t cap = 0;
        while (getline(&line, &cap, stdin) >= 0) {
            char *save = NULL; char *cmd = strtok_r(line, " \t\r\n", &save);
            if (!cmd) continue;
            if (!strcmp(cmd, "PING")) { printf("PONG\n"); continue; }
            if (!strcmp(cmd, "RESET")) { if (qn_qwen4_model_reset_session(model, err, sizeof(err)) || (mtp_enabled && qn_mtp_head_reset(mtp))) { printf("FATAL reset_failed %s\n", err); free(cached_tokens); free(line); qn_qwen4_model_close(model); return 3; } cached_n=0; cached_next_valid=0; printf("RESET\n"); continue; }
            if (!strcmp(cmd, "QUIT")) { printf("BYE\n"); break; }
            if (strcmp(cmd, "GEN")) { printf("ERR 400 unknown_command\n"); continue; }

            uint32_t max_tokens = 0, nstops = 0, nprompt = 0;
            char *x = next_tok(&save); if (parse_u32(x, &max_tokens) || !max_tokens || max_tokens > QN_WORKER_MAX_OUTPUT) { printf("ERR 400 invalid_max_tokens\n"); continue; }
            x = next_tok(&save); if (parse_u32(x, &nstops) || nstops > QN_WORKER_MAX_STOPS) { printf("ERR 400 invalid_stop_count\n"); continue; }
            uint32_t stops[QN_WORKER_MAX_STOPS]; int bad = 0;
            for (uint32_t i = 0; i < nstops; ++i) { x = next_tok(&save); if (parse_u32(x, &stops[i])) { bad = 1; break; } }
            if (bad) { printf("ERR 400 invalid_stop_id\n"); continue; }
            x = next_tok(&save); if (parse_u32(x, &nprompt) || !nprompt || nprompt > QN_WORKER_MAX_PROMPT) { printf("ERR 400 invalid_prompt_count\n"); continue; }
            if ((uint64_t)nprompt + (uint64_t)max_tokens > QN_WORKER_MAX_PROMPT) { printf("ERR 400 context_limit_exceeded\n"); continue; }
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
            if (!mtp_enabled) {
                for (;;) {
                    printf("TOK %u %.3f\n", tok, produced ? decode_ms : 0.0); produced++;
                    qn_qwen4_step_output so = {0};
                    if (qn_qwen4_model_step(model, tok, NULL, &so, err, sizeof(err))) { printf("FATAL decode_failed %s\n", err); free(cached_tokens); free(line); qn_qwen4_model_close(model); return 5; }
                    decode_ms += so.total_ms;
                    if (cached_n >= QN_WORKER_MAX_PROMPT) { printf("FATAL cache_context_overflow\n"); free(cached_tokens); free(line); qn_qwen4_model_close(model); return 6; }
                    cached_tokens[cached_n++] = tok; cached_next = so.next_token; cached_next_valid = 1;
                    if (is_stop(tok, stops, nstops)) { stopped = 1; break; }
                    if (produced >= max_tokens) break; tok = cached_next;
                }
            } else {
                if (qn_mtp_head_reset(mtp)) { printf("FATAL mtp_reset_failed\n"); free(cached_tokens); free(line); qn_qwen4_model_close(model); return 5; }
                qn_qwen4_model_stats mst={0}; qn_qwen4_model_get_stats(model,&mst); uint32_t mtp_base_pos=mst.position;
                float round_stream[10240]; if (qn_qwen4_model_copy_last_stream(model,round_stream)) { printf("FATAL mtp_last_stream_missing\n"); free(cached_tokens); free(line); qn_qwen4_model_close(model); return 5; }
                uint64_t mtp_attempts=0, mtp_accepts=0; double mtp_draft_ms=0, mtp_verify_ms=0, mtp_tail_ms=0, mtp_shadow_wait=0;
                while (produced < max_tokens) {
                    uint32_t remain=max_tokens-produced;
                    /* A known stop token or a one-token tail is cheaper/safer on the regular path. */
                    if (remain==1 || is_stop(tok,stops,nstops)) {
                        printf("TOK %u %.3f\n",tok,decode_ms); produced++; qn_qwen4_step_output so={0};
                        if(qn_qwen4_model_step(model,tok,NULL,&so,err,sizeof(err))){printf("FATAL decode_failed %s\n",err);free(cached_tokens);free(line);qn_qwen4_model_close(model);return 5;}
                        decode_ms+=so.total_ms;if(cached_n>=QN_WORKER_MAX_PROMPT){printf("FATAL cache_context_overflow\n");free(cached_tokens);free(line);qn_qwen4_model_close(model);return 6;}cached_tokens[cached_n++]=tok;cached_next=so.next_token;cached_next_valid=1;
                        if(is_stop(tok,stops,nstops))stopped=1;break;
                    }
                    uint32_t m=remain-1; if(m>5)m=5; uint32_t off0=qn_mtp_head_seq_offset(mtp),drafts[5]={0},pred[6]={0},vin[6]={0};
                    double round0=worker_ms(); if(qn_qwen4_model_spec_shadow_dispatch(model,err,sizeof(err))){printf("FATAL mtp_shadow_failed %s\n",err);free(cached_tokens);free(line);qn_qwen4_model_close(model);return 5;}
                    printf("TOK %u %.3f\n",tok,decode_ms); produced++;
                    float pstream[10240],dstream[10240];memcpy(pstream,round_stream,sizeof(pstream));uint32_t xdraft=tok;double d0=worker_ms();
                    for(uint32_t i=0;i<m;i++){qn_mtp_head_output mo={0};if(qn_mtp_head_forward(mtp,pstream,xdraft,mtp_base_pos+off0+i,dstream,&mo,err,sizeof(err))){printf("FATAL mtp_draft_failed %s\n",err);free(cached_tokens);free(line);qn_qwen4_model_close(model);return 5;}drafts[i]=mo.draft_token;memcpy(pstream,dstream,sizeof(pstream));xdraft=drafts[i];}
                    mtp_draft_ms+=worker_ms()-d0; double sw=0;if(qn_qwen4_model_spec_shadow_activate(model,&sw,err,sizeof(err))){printf("FATAL mtp_shadow_activate_failed %s\n",err);free(cached_tokens);free(line);qn_qwen4_model_close(model);return 5;}mtp_shadow_wait+=sw;
                    vin[0]=tok;for(uint32_t i=0;i<m;i++)vin[i+1]=drafts[i];float *streams=malloc((size_t)(m+1)*10240*sizeof(float));if(!streams){printf("FATAL allocation_failed\n");free(cached_tokens);free(line);qn_qwen4_model_close(model);return 70;}
                    qn_qwen4_prefill_output vo={0};double v0=worker_ms();if(qn_qwen4_model_verify_tokens_capture(model,vin,m+1,pred,streams,&vo,err,sizeof(err))){free(streams);printf("FATAL mtp_verify_failed %s\n",err);free(cached_tokens);free(line);qn_qwen4_model_close(model);return 5;}mtp_verify_ms+=worker_ms()-v0;
                    uint32_t a=0;while(a<m&&pred[a]==drafts[a])a++;mtp_attempts+=m;mtp_accepts+=a;
                    uint32_t emit_drafts=a;int stop_in_round=0;for(uint32_t i=0;i<a;i++)if(is_stop(drafts[i],stops,nstops)){emit_drafts=i+1;stop_in_round=1;break;}
                    if(a==m&&!stop_in_round){
                        if(qn_qwen4_model_spec_shadow_commit(model)){free(streams);printf("FATAL mtp_shadow_commit_failed\n");free(cached_tokens);free(line);qn_qwen4_model_close(model);return 5;}
                        if(cached_n>=QN_WORKER_MAX_PROMPT){free(streams);printf("FATAL cache_context_overflow\n");free(cached_tokens);free(line);qn_qwen4_model_close(model);return 6;}cached_tokens[cached_n++]=tok;
                        for(uint32_t i=0;i<m;i++){printf("TOK %u %.3f\n",drafts[i],decode_ms);if(cached_n>=QN_WORKER_MAX_PROMPT){free(streams);printf("FATAL cache_context_overflow\n");free(cached_tokens);free(line);qn_qwen4_model_close(model);return 6;}cached_tokens[cached_n++]=drafts[i];produced++;}
                        cached_next=pred[m];cached_next_valid=1;tok=cached_next;memcpy(round_stream,streams+(size_t)m*10240,sizeof(round_stream));
                        if(produced<max_tokens){double ht=0;const float *prev=streams+(size_t)(m-1)*10240;if(qn_mtp_head_append_history(mtp,prev,vin[m],mtp_base_pos+off0+m,&ht,err,sizeof(err))){free(streams);printf("FATAL mtp_tail_failed %s\n",err);free(cached_tokens);free(line);qn_qwen4_model_close(model);return 5;}mtp_tail_ms+=ht;}
                    }else{
                        if(qn_qwen4_model_spec_shadow_rollback(model,err,sizeof(err))){free(streams);printf("FATAL mtp_rollback_failed %s\n",err);free(cached_tokens);free(line);qn_qwen4_model_close(model);return 5;}
                        uint32_t commit_len=1+emit_drafts;qn_qwen4_prefill_output rp={0};if(qn_qwen4_model_prefill_tokens(model,vin,commit_len,NULL,&rp,err,sizeof(err))){free(streams);printf("FATAL mtp_replay_failed %s\n",err);free(cached_tokens);free(line);qn_qwen4_model_close(model);return 5;}
                        if(cached_n>=QN_WORKER_MAX_PROMPT){free(streams);printf("FATAL cache_context_overflow\n");free(cached_tokens);free(line);qn_qwen4_model_close(model);return 6;}cached_tokens[cached_n++]=tok;
                        for(uint32_t i=0;i<emit_drafts;i++){printf("TOK %u %.3f\n",drafts[i],decode_ms);if(cached_n>=QN_WORKER_MAX_PROMPT){free(streams);printf("FATAL cache_context_overflow\n");free(cached_tokens);free(line);qn_qwen4_model_close(model);return 6;}cached_tokens[cached_n++]=drafts[i];produced++;}
                        if(stop_in_round){stopped=1;cached_next=rp.next_token;cached_next_valid=1;free(streams);decode_ms+=worker_ms()-round0;break;}
                        cached_next=pred[a];cached_next_valid=1;tok=cached_next;if(qn_mtp_head_reset(mtp)||qn_qwen4_model_copy_last_stream(model,round_stream)){free(streams);printf("FATAL mtp_partial_reset_failed\n");free(cached_tokens);free(line);qn_qwen4_model_close(model);return 5;}qn_qwen4_model_get_stats(model,&mst);mtp_base_pos=mst.position;
                    }
                    free(streams);decode_ms+=worker_ms()-round0;if(stopped)break;
                }
                fprintf(stderr,"mtp_stats attempts=%llu accepts=%llu rate=%.3f draft_ms=%.3f verify_ms=%.3f tail_ms=%.3f shadow_wait_ms=%.3f\n",(unsigned long long)mtp_attempts,(unsigned long long)mtp_accepts,mtp_attempts?(double)mtp_accepts/mtp_attempts:0.0,mtp_draft_ms,mtp_verify_ms,mtp_tail_ms,mtp_shadow_wait);
            }
            printf("END %s %u %.3f\n", stopped ? "stop" : "length", produced, decode_ms);
        }
        free(cached_tokens); free(line); if(mtp)qn_mtp_head_close(mtp); if(mtp_io)qn_model_io_close(mtp_io); qn_qwen4_model_close(model); return 0;
    }
}
