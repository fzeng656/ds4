// Kimi Delta Attention kernels, adapted from the kimi-k3 branch.

struct glm53_kda_args {
    uint n_heads;
    uint n_rows;
    float lower_bound;
    float norm_eps;
};

/*
 * One threadgroup owns one (sequence, head). Four simdgroups update four
 * value rows concurrently; every lane owns four adjacent key columns.
 */
kernel void kernel_glm53_kda_decode(
        constant glm53_kda_args &args,
        device const float   *q_in,
        device const float   *k_in,
        device const float   *v_in,
        device const float   *raw_gate,
        device const float   *raw_beta,
        device const float   *output_gate,
        device const float   *q_conv,
        device const float   *k_conv,
        device const float   *v_conv,
        device const float   *a_log,
        device const float   *dt_bias,
        device const float   *output_norm,
        device float         *conv_state,
        device float         *state,
        device float         *out,
        threadgroup float    *scratch [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    constexpr uint HISTORY = 3u;
    const uint row = tgpig.x;
    const uint head = tgpig.y;
    if (row >= args.n_rows || head >= args.n_heads) return;

    threadgroup float *sq = scratch;
    threadgroup float *sk = sq + D;
    threadgroup float *sd = sk + D;
    threadgroup float *sv = sd + D;
    threadgroup float *so = sv + D;
    threadgroup float *reduce_q = so + D;
    threadgroup float *reduce_k = reduce_q + 4u;
    threadgroup float *reduce_o = reduce_k + 4u;
    threadgroup float *beta_shared = reduce_o + 4u;

    const uint projection = args.n_heads * D;
    const uint channel = head * D + tid;
    const ulong input_base = (ulong)row * projection + head * D;
    const ulong conv_row_stride = 3ul * HISTORY * projection;

    if (tid < D) {
        float q_acc = 0.0f;
        float k_acc = 0.0f;
        float v_acc = 0.0f;
        device float *q_state = conv_state +
            (ulong)row * conv_row_stride;
        device float *k_state = q_state + HISTORY * projection;
        device float *v_state = k_state + HISTORY * projection;
        for (uint w = 0; w < HISTORY; w++) {
            q_acc = fma(q_state[(ulong)w * projection + channel],
                        q_conv[(ulong)channel * 4u + w], q_acc);
            k_acc = fma(k_state[(ulong)w * projection + channel],
                        k_conv[(ulong)channel * 4u + w], k_acc);
            v_acc = fma(v_state[(ulong)w * projection + channel],
                        v_conv[(ulong)channel * 4u + w], v_acc);
        }
        const float q_new = q_in[input_base + tid];
        const float k_new = k_in[input_base + tid];
        const float v_new = v_in[input_base + tid];
        q_acc = fma(q_new, q_conv[(ulong)channel * 4u + 3u], q_acc);
        k_acc = fma(k_new, k_conv[(ulong)channel * 4u + 3u], k_acc);
        v_acc = fma(v_new, v_conv[(ulong)channel * 4u + 3u], v_acc);

        q_state[channel] = q_state[projection + channel];
        q_state[projection + channel] = q_state[2ul * projection + channel];
        q_state[2ul * projection + channel] = q_new;
        k_state[channel] = k_state[projection + channel];
        k_state[projection + channel] = k_state[2ul * projection + channel];
        k_state[2ul * projection + channel] = k_new;
        v_state[channel] = v_state[projection + channel];
        v_state[projection + channel] = v_state[2ul * projection + channel];
        v_state[2ul * projection + channel] = v_new;

        sq[tid] = q_acc / (1.0f + exp(-q_acc));
        sk[tid] = k_acc / (1.0f + exp(-k_acc));
        sv[tid] = v_acc / (1.0f + exp(-v_acc));
        const float gate = raw_gate[input_base + tid] + dt_bias[channel];
        sd[tid] = exp(args.lower_bound *
                      (1.0f / (1.0f + exp(-exp(a_log[head]) * gate))));
    }
    if (tid == 0u) {
        beta_shared[0] =
            1.0f / (1.0f + exp(-raw_beta[(ulong)row * args.n_heads + head]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup |
                       mem_flags::mem_device);

    float q_sumsq = sq[tid] * sq[tid];
    float k_sumsq = sk[tid] * sk[tid];
    q_sumsq = simd_sum(q_sumsq);
    k_sumsq = simd_sum(k_sumsq);
    if (lane == 0u) {
        reduce_q[sg] = q_sumsq;
        reduce_k[sg] = k_sumsq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q_total = lane < 4u ? reduce_q[lane] : 0.0f;
    float k_total = lane < 4u ? reduce_k[lane] : 0.0f;
    q_total = simd_sum(q_total);
    k_total = simd_sum(k_total);
    const float q_scale = rsqrt(q_total + 1.0e-6f) * 0x1.6a09e6p-4f;
    const float k_scale = rsqrt(k_total + 1.0e-6f);
    if (tid < D) {
        sq[tid] *= q_scale;
        sk[tid] *= k_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint k0 = lane * 4u;
    const float4 q4 = *((threadgroup float4 *)(sq + k0));
    const float4 k4 = *((threadgroup float4 *)(sk + k0));
    const float4 decay4 = *((threadgroup float4 *)(sd + k0));
    const ulong state_head =
        ((ulong)row * args.n_heads + head) * D * D;

    for (uint value = sg; value < D; value += 4u) {
        device float4 *hptr =
            (device float4 *)(state + state_head + (ulong)value * D + k0);
        float4 h = *hptr * decay4;
        float hk = dot(h, k4);
        hk = simd_sum(hk);
        const float delta_v = (sv[value] - hk) * beta_shared[0];
        h = fma(k4, float4(delta_v), h);
        *hptr = h;
        float hq = simd_sum(dot(h, q4));
        if (lane == 0u) so[value] = hq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup |
                       mem_flags::mem_device);

    float o_sumsq = so[tid] * so[tid];
    o_sumsq = simd_sum(o_sumsq);
    if (lane == 0u) reduce_o[sg] = o_sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o_total = lane < 4u ? reduce_o[lane] : 0.0f;
    o_total = simd_sum(o_total);
    const float o_scale = rsqrt(o_total / (float)D + args.norm_eps);
    if (tid < D) {
        const ulong index = input_base + tid;
        const float gate =
            1.0f / (1.0f + exp(-output_gate[index]));
        out[index] = so[tid] * o_scale * output_norm[tid] * gate;
    }
}

kernel void kernel_glm53_kda_prefill_prepare(
        constant glm53_kda_args &args,
        device float         *q,
        device float         *k,
        device float         *v,
        device float         *raw_gate,
        device const float   *q_conv,
        device const float   *k_conv,
        device const float   *v_conv,
        device const float   *a_log,
        device const float   *dt_bias,
        device float         *conv_state,
        threadgroup float    *scratch [[threadgroup(0)]],
        uint head [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    constexpr uint HISTORY = 3u;
    if (head >= args.n_heads) return;
    threadgroup float *sq = scratch;
    threadgroup float *sk = sq + D;
    threadgroup float *reduce_q = sk + D;
    threadgroup float *reduce_k = reduce_q + 4u;
    const uint projection = args.n_heads * D;
    const uint channel = head * D + tid;
    device float *q_state = conv_state;
    device float *k_state = q_state + HISTORY * projection;
    device float *v_state = k_state + HISTORY * projection;

    for (uint token = 0; token < args.n_rows; token++) {
        const ulong index = (ulong)token * projection + channel;
        float q_acc = 0.0f;
        float k_acc = 0.0f;
        float v_acc = 0.0f;
        for (uint w = 0; w < HISTORY; w++) {
            q_acc = fma(q_state[(ulong)w * projection + channel],
                        q_conv[(ulong)channel * 4u + w], q_acc);
            k_acc = fma(k_state[(ulong)w * projection + channel],
                        k_conv[(ulong)channel * 4u + w], k_acc);
            v_acc = fma(v_state[(ulong)w * projection + channel],
                        v_conv[(ulong)channel * 4u + w], v_acc);
        }
        const float q_new = q[index];
        const float k_new = k[index];
        const float v_new = v[index];
        q_acc = fma(q_new, q_conv[(ulong)channel * 4u + 3u], q_acc);
        k_acc = fma(k_new, k_conv[(ulong)channel * 4u + 3u], k_acc);
        v_acc = fma(v_new, v_conv[(ulong)channel * 4u + 3u], v_acc);
        q_state[channel] = q_state[projection + channel];
        q_state[projection + channel] = q_state[2ul * projection + channel];
        q_state[2ul * projection + channel] = q_new;
        k_state[channel] = k_state[projection + channel];
        k_state[projection + channel] = k_state[2ul * projection + channel];
        k_state[2ul * projection + channel] = k_new;
        v_state[channel] = v_state[projection + channel];
        v_state[projection + channel] = v_state[2ul * projection + channel];
        v_state[2ul * projection + channel] = v_new;

        sq[tid] = q_acc / (1.0f + exp(-q_acc));
        sk[tid] = k_acc / (1.0f + exp(-k_acc));
        v[index] = v_acc / (1.0f + exp(-v_acc));
        const float gate = raw_gate[index] + dt_bias[channel];
        raw_gate[index] = exp(args.lower_bound *
            (1.0f / (1.0f + exp(-exp(a_log[head]) * gate))));
        threadgroup_barrier(mem_flags::mem_threadgroup |
                           mem_flags::mem_device);

        float q_sumsq = simd_sum(sq[tid] * sq[tid]);
        float k_sumsq = simd_sum(sk[tid] * sk[tid]);
        if (lane == 0u) {
            reduce_q[sg] = q_sumsq;
            reduce_k[sg] = k_sumsq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float q_total = lane < 4u ? reduce_q[lane] : 0.0f;
        float k_total = lane < 4u ? reduce_k[lane] : 0.0f;
        q_total = simd_sum(q_total);
        k_total = simd_sum(k_total);
        q[index] = sq[tid] * rsqrt(q_total + 1.0e-6f) *
                   0x1.6a09e6p-4f;
        k[index] = sk[tid] * rsqrt(k_total + 1.0e-6f);
        threadgroup_barrier(mem_flags::mem_threadgroup |
                           mem_flags::mem_device);
    }
}

kernel void kernel_glm53_kda_prefill_recurrence(
        constant glm53_kda_args &args,
        device const float   *q,
        device const float   *k,
        device const float   *v,
        device const float   *decay,
        device const float   *raw_beta,
        device float         *state,
        device float         *out,
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    const uint head = tgpig.x;
    const uint value = tgpig.y * 4u + sg;
    if (head >= args.n_heads || value >= D) return;
    const uint projection = args.n_heads * D;
    const uint k0 = lane * 4u;
    device float4 *state_ptr = (device float4 *)(
        state + ((ulong)head * D + value) * D + k0);
    float4 h = *state_ptr;

    for (uint token = 0; token < args.n_rows; token++) {
        const ulong base = (ulong)token * projection + head * D;
        const float4 q4 = *((device const float4 *)(q + base + k0));
        const float4 k4 = *((device const float4 *)(k + base + k0));
        const float4 decay4 =
            *((device const float4 *)(decay + base + k0));
        h *= decay4;
        const float hk = simd_sum(dot(h, k4));
        const float beta = 1.0f /
            (1.0f + exp(-raw_beta[(ulong)token * args.n_heads + head]));
        const float delta_v = (v[base + value] - hk) * beta;
        h = fma(k4, float4(delta_v), h);
        const float result = simd_sum(dot(h, q4));
        if (lane == 0u) out[base + value] = result;
    }
    *state_ptr = h;
}

kernel void kernel_glm53_kda_prefill_recurrence_percore120(
        constant glm53_kda_args &args,
        device const float   *q,
        device const float   *k,
        device const float   *v,
        device const float   *decay,
        device const float   *raw_beta,
        device float         *state,
        device float         *out,
        threadgroup float    *scratch [[threadgroup(0)]],
        uint tg [[threadgroup_position_in_grid]],
        ushort tid_u [[thread_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    constexpr uint TB = 8u;
    constexpr uint NTG = 120u;
    constexpr uint MR = 70u;
    constexpr uint ROWS_PER_GROUP = 2u;
    constexpr uint THREADS = 280u;

    const uint tid = uint(tid_u);
    const uint total = args.n_heads * D;
    if (tg >= NTG || total == 0u) return;

    // Even boundaries guarantee a two-row register pair never straddles a
    // 128-row head boundary.
    const uint half_total = total / 2u;
    const uint r0 = 2u * ((tg * half_total) / NTG);
    const uint r1 = 2u * (((tg + 1u) * half_total) / NTG);
    const uint nrows = r1 - r0;
    const uint head_a = r0 / D;
    const uint head_z = (r1 - 1u) / D;
    const uint nh = head_z - head_a + 1u;

    const uint rg = tid / 8u;
    const uint seg = tid & 7u;
    const uint d0 = seg * 16u;
    const uint lr0 = rg * ROWS_PER_GROUP;
    const uint lr1 = lr0 + 1u;
    const bool active0 = lr0 < nrows;
    const bool active1 = lr1 < nrows;
    const uint fr0 = r0 + min(lr0, nrows - 1u);
    const uint fr1 = r0 + min(lr1, nrows - 1u);
    const uint h0 = fr0 / D;
    const uint h1 = fr1 / D;
    const uint v0 = fr0 % D;
    const uint v1 = fr1 % D;
    const uint hs0 = h0 - head_a;
    const uint hs1 = h1 - head_a;

    threadgroup float *sq = scratch;
    threadgroup float *sk = sq + 2u * TB * D;
    threadgroup float *sd = sk + 2u * TB * D;
    threadgroup float *sv = sd + 2u * TB * D;
    threadgroup float *sb = sv + TB * MR;

    const uint projection = args.n_heads * D;

    float4 h00=0.0f,h01=0.0f,h02=0.0f,h03=0.0f;
    float4 h10=0.0f,h11=0.0f,h12=0.0f,h13=0.0f;
    device float4 *s0 = nullptr;
    device float4 *s1 = nullptr;
    if (active0) {
        s0 = (device float4 *)(state + (ulong)fr0 * D + d0);
        h00=s0[0]; h01=s0[1]; h02=s0[2]; h03=s0[3];
    }
    if (active1) {
        s1 = (device float4 *)(state + (ulong)fr1 * D + d0);
        h10=s1[0]; h11=s1[1]; h12=s1[2]; h13=s1[3];
    }

    for (uint t0=0u; t0<args.n_rows; t0+=TB) {
        const uint nt=min(TB,args.n_rows-t0);

        for (uint hs=0u; hs<nh; ++hs) {
            const uint head=head_a+hs;
            for (uint p=tid; p<nt*D; p+=THREADS) {
                const uint tr=p/D;
                const uint d=p-tr*D;
                const ulong base=(ulong)(t0+tr)*projection+(ulong)head*D;
                const ulong dst=((ulong)hs*TB+tr)*D+d;
                sq[dst]=q[base+d];
                sk[dst]=k[base+d];
                sd[dst]=decay[base+d];
            }
            for (uint tr=tid; tr<nt; tr+=THREADS) {
                sb[hs*TB+tr]=1.0f/
                    (1.0f+exp(-raw_beta[(ulong)(t0+tr)*args.n_heads+head]));
            }
        }

        for (uint p=tid; p<nt*nrows; p+=THREADS) {
            const uint tr=p/nrows;
            const uint lr=p-tr*nrows;
            const uint fr=r0+lr;
            const uint head=fr/D;
            const uint value=fr%D;
            const ulong base=(ulong)(t0+tr)*projection+(ulong)head*D;
            sv[tr*MR+lr]=v[base+value];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint tr=0u; tr<nt; ++tr) {
            if (active0) {
                const threadgroup float4 *q4=(const threadgroup float4 *)
                    (sq+((ulong)hs0*TB+tr)*D+d0);
                const threadgroup float4 *k4=(const threadgroup float4 *)
                    (sk+((ulong)hs0*TB+tr)*D+d0);
                const threadgroup float4 *g4=(const threadgroup float4 *)
                    (sd+((ulong)hs0*TB+tr)*D+d0);
                const float4 q0=q4[0],q1=q4[1],q2=q4[2],q3=q4[3];
                const float4 k0=k4[0],k1=k4[1],k2=k4[2],k3=k4[3];
                h00*=g4[0]; h01*=g4[1]; h02*=g4[2]; h03*=g4[3];
                float pp=dot(h00,k0)+dot(h01,k1)+dot(h02,k2)+dot(h03,k3);
                pp+=simd_shuffle_xor(pp,4);
                pp+=simd_shuffle_xor(pp,2);
                pp+=simd_shuffle_xor(pp,1);
                const float dv=(sv[tr*MR+lr0]-pp)*sb[hs0*TB+tr];
                h00=fma(k0,float4(dv),h00);
                h01=fma(k1,float4(dv),h01);
                h02=fma(k2,float4(dv),h02);
                h03=fma(k3,float4(dv),h03);
                float yy=dot(h00,q0)+dot(h01,q1)+dot(h02,q2)+dot(h03,q3);
                yy+=simd_shuffle_xor(yy,4);
                yy+=simd_shuffle_xor(yy,2);
                yy+=simd_shuffle_xor(yy,1);
                if (seg==0u) {
                    const ulong base=(ulong)(t0+tr)*projection+(ulong)h0*D;
                    out[base+v0]=yy;
                }
            }
            if (active1) {
                const threadgroup float4 *q4=(const threadgroup float4 *)
                    (sq+((ulong)hs1*TB+tr)*D+d0);
                const threadgroup float4 *k4=(const threadgroup float4 *)
                    (sk+((ulong)hs1*TB+tr)*D+d0);
                const threadgroup float4 *g4=(const threadgroup float4 *)
                    (sd+((ulong)hs1*TB+tr)*D+d0);
                const float4 q0=q4[0],q1=q4[1],q2=q4[2],q3=q4[3];
                const float4 k0=k4[0],k1=k4[1],k2=k4[2],k3=k4[3];
                h10*=g4[0]; h11*=g4[1]; h12*=g4[2]; h13*=g4[3];
                float pp=dot(h10,k0)+dot(h11,k1)+dot(h12,k2)+dot(h13,k3);
                pp+=simd_shuffle_xor(pp,4);
                pp+=simd_shuffle_xor(pp,2);
                pp+=simd_shuffle_xor(pp,1);
                const float dv=(sv[tr*MR+lr1]-pp)*sb[hs1*TB+tr];
                h10=fma(k0,float4(dv),h10);
                h11=fma(k1,float4(dv),h11);
                h12=fma(k2,float4(dv),h12);
                h13=fma(k3,float4(dv),h13);
                float yy=dot(h10,q0)+dot(h11,q1)+dot(h12,q2)+dot(h13,q3);
                yy+=simd_shuffle_xor(yy,4);
                yy+=simd_shuffle_xor(yy,2);
                yy+=simd_shuffle_xor(yy,1);
                if (seg==0u) {
                    const ulong base=(ulong)(t0+tr)*projection+(ulong)h1*D;
                    out[base+v1]=yy;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active0) { s0[0]=h00; s0[1]=h01; s0[2]=h02; s0[3]=h03; }
    if (active1) { s1[0]=h10; s1[1]=h11; s1[2]=h12; s1[3]=h13; }
}

kernel void kernel_glm53_kda_prefill_output(
        constant glm53_kda_args &args,
        device float         *out,
        device const float   *output_gate,
        device const float   *output_norm,
        threadgroup float    *partial [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    const uint token = tgpig.x;
    const uint head = tgpig.y;
    if (token >= args.n_rows || head >= args.n_heads) return;
    const uint projection = args.n_heads * D;
    const ulong base = (ulong)token * projection + head * D;
    const float raw = out[base + tid];
    float sumsq = simd_sum(raw * raw);
    if (lane == 0u) partial[sg] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = lane < 4u ? partial[lane] : 0.0f;
    total = simd_sum(total);
    const float scale = rsqrt(total / (float)D + args.norm_eps);
    out[base + tid] = raw * scale * output_norm[tid] /
        (1.0f + exp(-output_gate[base + tid]));
}
