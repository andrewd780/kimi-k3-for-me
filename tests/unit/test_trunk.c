/* test_trunk.c - streaming trunk regression test.
 *
 * WHY THIS FILE EXISTS
 *   The trunk streams layers through a ring buffer with optional async prefetch.
 *   When the ring has only one slot, starting the async reader thread is a
 *   correctness bug: k3_trunk_prefetch claims the ring slot for the incoming
 *   layer, the reader preads layer L+1 straight over layer L's bytes while the
 *   caller is still computing on them, and nothing detects it -- the run
 *   completes and emits fluent, wrong tokens.
 *
 *   The one-slot guard in k3_trunk_open prevents the reader thread from starting
 *   when the ring has fewer than two slots. This test proves the guard works, verifies prefetch correctness,
 *   checks slot isolation with two slots, exercises ring wrap-around, and ensures
 *   failed prefetches are never published.
 *
 * WHAT IS CHECKED
 *   1  GUARD        with a budget forcing one slot, io_state is NULL. A build that
 *                   removed the guard would fail here because io_state != NULL.
 *   2  NO-OP        prefetch with one slot is a no-op; the active layer survives.
 *   3  ACCOUNTING   misses/hits match the expected synchronous path.
 *   4  ISOLATION    with two slots, a prefetched read lands in a different slot
 *                   from the active layer. Both layers' bound pointers stay correct.
 *   5  RING-WRAP    binding three layers through two slots forces eviction and
 *                   proves the ring correctly recycles slots.
 *   6  FAILED-READ  truncating trunk.bin after open causes a prefetch to fail;
 *                   the slot is never published and bind returns an error.
 *   7  ROWS         the --trunk-rows pipeline (k3_trunk_open_rows): two full walks and
 *                   a batch of positions give products bitwise equal to the resident
 *                   trunk's, a batch reads each matrix once, a failed read is sticky and
 *                   an undersized budget is refused.
 *
 * usage: test_trunk
 *   writes a synthetic trunk fixture to a temp directory, then drives k3_trunk_open /
 *   bind / prefetch / close at two budgets and the row pipeline at one. Built with
 *   -DK3_TEST_ROWS_ONLY (bin/test_trunk_rows) it runs check 7 alone on 93 layers whose
 *   257-row dense matrices span several 4 KiB row tiles, the case where the pipeline's
 *   double buffering, tile offsets and O_DIRECT prefixes are actually exercised; in the
 *   3-layer build every matrix fits in a single tile.
 */
#define _GNU_SOURCE            /* O_DIRECT, mkdtemp, ftruncate */
#define _POSIX_C_SOURCE 200809L

#include "k3_portable_io.h"   /* first: establishes Darwin feature macros */

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>            /* ftruncate */

#include "k3.h"
#include "k3_bind.h"
#include "k3_trunk.h"

#define PRE         "language_model.model."
#define HIDDEN      16
#define KDA_HEADS   2
#define KDA_HEAD_DIM 4    /* A_log ships 4 elements, engine takes first 2 */
#define CONV_K      2
#ifdef K3_TEST_ROWS_ONLY
#define DENSE_INTER 257       /* ragged rows across multiple 4 KiB tiles */
#define N_LAYERS    93
#define FIXTURE_RUN_BYTES 32768
#else
#define DENSE_INTER 16
#define N_LAYERS    3
#define FIXTURE_RUN_BYTES 4096
#endif
#define N_EXPERTS   1
#define LATENT      4
#define N_SHARED    1
#define MOE_INTER   8

static int g_fail = 0;

static void ck(int ok, const char *what, const char *detail)
{
    printf("  %s  %-34s %s\n", ok ? "PASS" : "FAIL", what, detail ? detail : "");
    if (!ok) g_fail++;
}

/* Byte i of tensor (layer, tensor_index): a hash of all three, so no two rows of a matrix
 * are alike and no two tensors share bytes.
 *
 * WHY EVERY BYTE DIFFERS. This fixture used to fill each tensor with one repeated byte.
 * Every row of every matrix was then the same, every output of a product had the same
 * value, and a row tile read from the wrong offset, applied to the wrong output rows or
 * shifted by a wrong O_DIRECT prefix produced exactly the right answer: the row-pipeline
 * checks below compared equal floats whatever the pipeline did. With a different byte at
 * every position, any such error moves some output and memcmp sees it.
 *
 * Odd bytes are the high byte of a BF16 value (and bytes 1 and 3 of an F32 one). They keep
 * a hashed sign and take an exponent near 1 (0x3A..0x3F), so every weight and vector is
 * finite and moderate: no NaN or infinity can make two different products compare the
 * same, and no sum overflows. */
static unsigned char fixture_byte(int layer, int tensor_index, int i)
{
    uint32_t h = (uint32_t)layer * 0x9E3779B1u ^ (uint32_t)tensor_index * 0x85EBCA77u ^
                 (uint32_t)i * 0xC2B2AE3Du;
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    if (!(i & 1)) return (unsigned char)h;
    return (unsigned char)((h & 0x80u) | (0x3Au + (h >> 8) % 6u));
}

/* Verify that the first n bytes at raw are tensor (layer, tensor_index)'s fixture bytes. */
static int check_marker(const unsigned char *raw, int layer, int tensor_index, int n)
{
    for (int i = 0; i < n; i++)
        if (raw[i] != fixture_byte(layer, tensor_index, i)) return 0;
    return 1;
}

/* Fill n bytes at dst with tensor (layer, tensor_index)'s fixture bytes. */
static void fill_marker(unsigned char *dst, int layer, int tensor_index, int n)
{
    for (int i = 0; i < n; i++) dst[i] = fixture_byte(layer, tensor_index, i);
}

/* Lay out one layer's tensors into dst, returning the number of bytes written.
 * Writes them in the plan_layer evaluation order so offsets are deterministic.
 * is_dense: 1 for the dense MLP, 0 for MoE layers.
 * is_mla:   1 for MLA attention layers, 0 for KDA.
 * ti        pointer to running tensor_index counter. */
static int gen_layer_run(unsigned char *dst, int layer, int is_dense, int is_mla, int *ti)
{
    const int H = HIDDEN;
    const int P = KDA_HEADS * KDA_HEAD_DIM;   /* 8 */
    unsigned char *p = dst;
    /* ---- 6 norms, always present, all F32 ---- */
    for (int i = 0; i < 6; i++, (*ti)++) {
        int nb = H * 4;  /* [H] F32 */
        fill_marker(p, layer, *ti, nb);
        p += nb;
    }

    if (is_mla) {
        /* ---- MLA attention ---- */
        const int qh = 6;   /* qk_nope=4 + qk_rope=2 */
        const int q_lora = 4, kv_lora = 4;
        const int n_heads = 2;
        /* q_a */
        { int nb = q_lora * H * 2; fill_marker(p, layer, (*ti)++, nb); p += nb; }
        /* q_a_norm */
        { int nb = q_lora * 4; fill_marker(p, layer, (*ti)++, nb); p += nb; }
        /* q_b */
        { int nb = n_heads * qh * q_lora * 2; fill_marker(p, layer, (*ti)++, nb); p += nb; }
        /* kv_a */
        { int nb = (kv_lora + 2) * H * 2; fill_marker(p, layer, (*ti)++, nb); p += nb; }
        /* kv_a_norm */
        { int nb = kv_lora * 4; fill_marker(p, layer, (*ti)++, nb); p += nb; }
        /* kv_b */
        { int nb = n_heads * (4 + 4) * kv_lora * 2; fill_marker(p, layer, (*ti)++, nb); p += nb; }
        /* o */
        { int nb = H * n_heads * 4 * 2; fill_marker(p, layer, (*ti)++, nb); p += nb; }
        /* g is skipped when mla_out_gate=0 */
    } else {
        /* ---- KDA attention ---- */
        /* q_proj */
        { int nb = P * H * 2; fill_marker(p, layer, (*ti)++, nb); p += nb; }
        /* k_proj */
        { int nb = P * H * 2; fill_marker(p, layer, (*ti)++, nb); p += nb; }
        /* v_proj */
        { int nb = P * H * 2; fill_marker(p, layer, (*ti)++, nb); p += nb; }
        /* g_proj */
        { int nb = P * H * 2; fill_marker(p, layer, (*ti)++, nb); p += nb; }
        /* o_proj */
        { int nb = H * P * 2; fill_marker(p, layer, (*ti)++, nb); p += nb; }
        /* q_conv1d */
        { int nb = P * CONV_K * 4; fill_marker(p, layer, (*ti)++, nb); p += nb; }
        /* k_conv1d */
        { int nb = P * CONV_K * 4; fill_marker(p, layer, (*ti)++, nb); p += nb; }
        /* v_conv1d */
        { int nb = P * CONV_K * 4; fill_marker(p, layer, (*ti)++, nb); p += nb; }
        /* f_a_proj */
        { int nb = KDA_HEAD_DIM * H * 2; fill_marker(p, layer, (*ti)++, nb); p += nb; }
        /* f_b_proj */
        { int nb = P * KDA_HEAD_DIM * 2; fill_marker(p, layer, (*ti)++, nb); p += nb; }
        /* b_proj */
        { int nb = KDA_HEADS * H * 2; fill_marker(p, layer, (*ti)++, nb); p += nb; }
        /* A_log -- 4 F32 elements, 16 bytes */
        { int nb = KDA_HEAD_DIM * 4; fill_marker(p, layer, (*ti)++, nb); p += nb; }
        /* dt_bias */
        { int nb = P * 4; fill_marker(p, layer, (*ti)++, nb); p += nb; }
        /* o_norm */
        { int nb = KDA_HEAD_DIM * 4; fill_marker(p, layer, (*ti)++, nb); p += nb; }
    }

    if (is_dense) {
        /* ---- dense MLP ---- */
        { int nb = DENSE_INTER * H * 2; fill_marker(p, layer, (*ti)++, nb); p += nb; }
        { int nb = DENSE_INTER * H * 2; fill_marker(p, layer, (*ti)++, nb); p += nb; }
        { int nb = H * DENSE_INTER * 2; fill_marker(p, layer, (*ti)++, nb); p += nb; }
    } else {
        /* ---- MoE ---- */
        /* gate */
        { int nb = N_EXPERTS * H * 4; fill_marker(p, layer, (*ti)++, nb); p += nb; }
        /* bias */
        { int nb = N_EXPERTS * 4; fill_marker(p, layer, (*ti)++, nb); p += nb; }
        /* down */
        { int nb = LATENT * H * 2; fill_marker(p, layer, (*ti)++, nb); p += nb; }
        /* up */
        { int nb = H * LATENT * 2; fill_marker(p, layer, (*ti)++, nb); p += nb; }
        /* latent_norm */
        { int nb = LATENT * 4; fill_marker(p, layer, (*ti)++, nb); p += nb; }
        /* sh1 */
        { int nb = MOE_INTER * N_SHARED * H * 2; fill_marker(p, layer, (*ti)++, nb); p += nb; }
        /* sh3 */
        { int nb = MOE_INTER * N_SHARED * H * 2; fill_marker(p, layer, (*ti)++, nb); p += nb; }
        /* sh2 */
        { int nb = H * MOE_INTER * N_SHARED * 2; fill_marker(p, layer, (*ti)++, nb); p += nb; }
    }

    return (int)(p - dst);
}

/* Write the tensor name for a given suffix and layer into buf.
 * Returns 0 on success, -1 if truncated. */
static int tensor_name(char *buf, size_t sz, int layer, const char *suffix)
{
    int n = snprintf(buf, sz, PRE "layers.%d.%s", layer, suffix);
    return (n > 0 && (size_t)n < sz) ? 0 : -1;
}

typedef struct {
    const char *suffix;
    int nbytes;
    int is_f32;   /* 1=F32, 0=BF16 */
} TensorDef;

/* Fill defs with the tensor descriptors for one layer, in gen_layer_run order.
 * Returns number of tensors. */
static int layer_tensor_defs(TensorDef *defs, int max, int is_dense, int is_mla)
{
    int n = 0;
    /* 6 norms */
    static const char *norm_names[] = {
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attention_res_norm.weight",
        "self_attention_res_proj.weight",
        "mlp_res_norm.weight",
        "mlp_res_proj.weight",
    };
    for (int i = 0; i < 6 && n < max; i++)
        { defs[n].suffix = norm_names[i]; defs[n].nbytes = HIDDEN * 4; defs[n].is_f32 = 1; n++; }

    if (is_mla) {
        TensorDef mla[] = {
            {"self_attn.q_a_proj.weight",               4 * HIDDEN * 2,   0},
            {"self_attn.q_a_layernorm.weight",          4 * 4,            1},
            {"self_attn.q_b_proj.weight",               2 * 6 * 4 * 2,    0},
            {"self_attn.kv_a_proj_with_mqa.weight",     (4+2) * HIDDEN * 2, 0},
            {"self_attn.kv_a_layernorm.weight",         4 * 4,            1},
            {"self_attn.kv_b_proj.weight",              2 * (4+4) * 4 * 2, 0},
            {"self_attn.o_proj.weight",                 HIDDEN * 2 * 4 * 2, 0},
        };
        int mn = (int)(sizeof mla / sizeof mla[0]);
        for (int i = 0; i < mn && n < max; i++) defs[n++] = mla[i];
        /* g_proj skipped: mla_out_gate=0 */
    } else {
        int P = KDA_HEADS * KDA_HEAD_DIM;
        TensorDef kda[] = {
            {"self_attn.q_proj.weight",     P * HIDDEN * 2,        0},
            {"self_attn.k_proj.weight",     P * HIDDEN * 2,        0},
            {"self_attn.v_proj.weight",     P * HIDDEN * 2,        0},
            {"self_attn.g_proj.weight",     P * HIDDEN * 2,        0},
            {"self_attn.o_proj.weight",     HIDDEN * P * 2,        0},
            {"self_attn.q_conv1d.weight",   P * CONV_K * 4,        1},
            {"self_attn.k_conv1d.weight",   P * CONV_K * 4,        1},
            {"self_attn.v_conv1d.weight",   P * CONV_K * 4,        1},
            {"self_attn.f_a_proj.weight",   KDA_HEAD_DIM * HIDDEN * 2, 0},
            {"self_attn.f_b_proj.weight",   P * KDA_HEAD_DIM * 2, 0},
            {"self_attn.b_proj.weight",     KDA_HEADS * HIDDEN * 2, 0},
            {"self_attn.A_log",             KDA_HEAD_DIM * 4,      1},
            {"self_attn.dt_bias",           P * 4,                 1},
            {"self_attn.o_norm.weight",     KDA_HEAD_DIM * 4,      1},
        };
        int kn = (int)(sizeof kda / sizeof kda[0]);
        for (int i = 0; i < kn && n < max; i++) defs[n++] = kda[i];
    }

    if (is_dense) {
        TensorDef dense[] = {
            {"mlp.gate_proj.weight",  DENSE_INTER * HIDDEN * 2, 0},
            {"mlp.up_proj.weight",    DENSE_INTER * HIDDEN * 2, 0},
            {"mlp.down_proj.weight",  HIDDEN * DENSE_INTER * 2, 0},
        };
        for (int i = 0; i < 3 && n < max; i++) defs[n++] = dense[i];
    } else {
        int SI = MOE_INTER * N_SHARED;
        TensorDef moe[] = {
            {"block_sparse_moe.gate.weight",                  N_EXPERTS * HIDDEN * 4, 1},
            {"block_sparse_moe.gate.e_score_correction_bias", N_EXPERTS * 4,          1},
            {"block_sparse_moe.routed_expert_down_proj.weight", LATENT * HIDDEN * 2, 0},
            {"block_sparse_moe.routed_expert_up_proj.weight",   HIDDEN * LATENT * 2, 0},
            {"block_sparse_moe.routed_expert_norm.weight",      LATENT * 4,            1},
            {"block_sparse_moe.shared_experts.gate_proj.weight", SI * HIDDEN * 2,     0},
            {"block_sparse_moe.shared_experts.up_proj.weight",   SI * HIDDEN * 2,     0},
            {"block_sparse_moe.shared_experts.down_proj.weight", HIDDEN * SI * 2,     0},
        };
        for (int i = 0; i < 8 && n < max; i++) defs[n++] = moe[i];
    }

    return n;
}

/* Write trunk.bin and trunk.json to dir. Returns 0 on success. */
static int gen_fixture(const char *dir)
{
    char path[1024];
    snprintf(path, sizeof path, "%s/trunk.bin", dir);
    FILE *f = fopen(path, "wb");
    if (!f) { perror("trunk.bin"); return -1; }

    unsigned char buf[FIXTURE_RUN_BYTES];
    int64_t file_off = 0;
    int nlayers = N_LAYERS;

    for (int L = 0; L < nlayers; L++) {
        memset(buf, 0, sizeof buf);
        int ti = 0;
        int nw = gen_layer_run(buf, L, L == 0, L % 3 == 2, &ti);
        if (nw <= 0 || nw > FIXTURE_RUN_BYTES) { fclose(f); return -1; }
        /* Write the full 4096-byte slot: raw[L] valid bytes + zero tail. */
        size_t wrote = 0;
        while (wrote < FIXTURE_RUN_BYTES) {
            size_t chunk = fwrite(buf + wrote, 1, FIXTURE_RUN_BYTES - wrote, f);
            if (chunk == 0) { fclose(f); return -1; }
            wrote += chunk;
        }
        file_off += FIXTURE_RUN_BYTES;
    }
    fclose(f);

    /* Write trunk.json */
    snprintf(path, sizeof path, "%s/trunk.json", dir);
    FILE *jf = fopen(path, "w");
    if (!jf) { perror("trunk.json"); return -1; }

    fprintf(jf, "{\n  \"n_layers\": %d,\n  \"align\": 4096,\n  \"layers\": [\n", nlayers);
    file_off = 0;
    for (int L = 0; L < nlayers; L++) {
        fprintf(jf, "    {\n      \"layer\": %d,\n      \"file_off\": %lld,"
                      "\n      \"nbytes\": %lld,\n      \"tensors\": {\n",
                L, (long long)file_off, (long long)FIXTURE_RUN_BYTES);

        TensorDef defs[36];
        int nd = layer_tensor_defs(defs, 36, L == 0, L % 3 == 2);

        int64_t off = 0;
        for (int t = 0; t < nd; t++) {
            char name[256];
            if (tensor_name(name, sizeof name, L, defs[t].suffix) != 0)
                { fclose(jf); return -1; }
            const char *dt = defs[t].is_f32 ? "F32" : "BF16";
            fprintf(jf, "        \"%s\": {\"off\":%lld,\"nbytes\":%d,\"dtype\":\"%s\"}%s\n",
                    name, (long long)off, defs[t].nbytes, dt,
                    t + 1 < nd ? "," : "");
            off += defs[t].nbytes;
        }

        fprintf(jf, "      }\n    }%s\n",
                L + 1 < nlayers ? "," : "");
        file_off += FIXTURE_RUN_BYTES;
    }
    fprintf(jf, "  ]\n}\n");
    fclose(jf);
    return 0;
}

/* ---- test entry points ---- */

/* One-slot: budget=12288 forces nslot=1, io_state=NULL */
static int test_one_slot(const char *dir, const K3Cfg *c)
{
    K3Trunk tr;
    if (k3_trunk_open(&tr, dir, c, 12288) != 0) {
        fprintf(stderr, "TRUNK OPEN FAILED (one-slot)\n"); return 1;
    }
    ck(tr.nslot == 1, "one-slot: ring is 1 slot", "");
    ck(tr.npin == 0,  "one-slot: nothing pinned", "");

    int guard_ok = (tr.io_state == NULL);
    ck(guard_ok, "one-slot: io_state is NULL (guard)", "");
    if (!guard_ok) {
        fprintf(stderr, "  *** the one-slot reader guard did not fire. "
                        "A build without it silently corrupts active layers.\n");
    }

    K3LayerBind b0; memset(&b0, 0, sizeof b0);
    if (k3_trunk_bind(&tr, c, 0, &b0) != 0) { k3_trunk_close(&tr); return 1; }
    ck(tr.hits == 0 && tr.misses == 1, "after bind L0: misses=1, hits=0", "");

    /* dt_bias is F32, pointed directly at run+off, 8×4=32 bytes, ti=18 in KDA path */
    int dt_ti = 18;   /* dt_bias is the 19th tensor for KDA (0-indexed 18) */
    int dt_ok_0 = check_marker((const unsigned char *)b0.kda.dt_bias, 0, dt_ti, 32);
    ck(dt_ok_0, "one-slot: bind L0 content correct", "");

    /* Prefetch L1 with one slot -- must be a no-op */
    k3_trunk_prefetch(&tr, 1);
    /* Prefetch was a no-op: misses should still be 1 (only the sync read of L0) */
    ck(tr.misses == 1, "one-slot: prefetch L1 no-op (misses unchanged)", "");

    /* Verify L0 is STILL intact after the no-op prefetch */
    int dt_ok_0b = check_marker((const unsigned char *)b0.kda.dt_bias, 0, dt_ti, 32);
    ck(dt_ok_0b, "one-slot: L0 survives prefetch L1", "");

    /* Now bind L1 -- this overwrites slot 0 legitimately */
    K3LayerBind b1; memset(&b1, 0, sizeof b1);
    if (k3_trunk_bind(&tr, c, 1, &b1) != 0) { k3_trunk_close(&tr); return 1; }
    ck(tr.hits == 0 && tr.misses == 2, "after bind L1: misses=2, hits=0", "");

    /* rebind L1 -- should be a hit now */
    int dt_ok_1 = check_marker((const unsigned char *)b1.kda.dt_bias, 1, dt_ti, 32);
    ck(dt_ok_1, "one-slot: bind L1 content correct", "");
    if (k3_trunk_bind(&tr, c, 1, &b1) != 0) { k3_trunk_close(&tr); return 1; }
    ck(tr.hits == 1 && tr.misses == 2, "after rebind L1: misses=2, hits=1", "");

    k3_trunk_close(&tr);
    return 0;
}

/* Two-slot: budget=24576 forces nslot=2, io_state!=NULL. Tests isolation. */
static int test_two_slot(const char *dir, const K3Cfg *c)
{
    K3Trunk tr;
    if (k3_trunk_open(&tr, dir, c, 24576) != 0) {
        fprintf(stderr, "TRUNK OPEN FAILED (two-slot)\n"); return 1;
    }
    ck(tr.nslot == 2, "two-slot: ring has 2 slots", "");
    ck(tr.io_state != NULL, "two-slot: io_state is not NULL", "");

    int dt_ti = 18;

    /* Bind L0 */
    K3LayerBind b0; memset(&b0, 0, sizeof b0);
    if (k3_trunk_bind(&tr, c, 0, &b0) != 0) { k3_trunk_close(&tr); return 1; }
    ck(tr.hits == 0 && tr.misses == 1, "two-slot: after bind L0, misses=1", "");

    int dt_ok_0 = check_marker((const unsigned char *)b0.kda.dt_bias, 0, dt_ti, 32);
    ck(dt_ok_0, "two-slot: bind L0 content correct", "");

    /* Prefetch L1 -- starts async read into slot 1 */
    k3_trunk_prefetch(&tr, 1);

    /* Bind L1 -- blocks until reader finishes, publishes slot 1.
     * The miss was recorded inside trunk_io_wait when it published the slot;
     * k3_trunk_bind itself records a hit only on the synchronous lookup path. */
    K3LayerBind b1; memset(&b1, 0, sizeof b1);
    if (k3_trunk_bind(&tr, c, 1, &b1) != 0) { k3_trunk_close(&tr); return 1; }
    ck(tr.hits == 0 && tr.misses == 2, "two-slot: after bind L1, misses=2", "");

    /* Verify both layers' pointer content AFTER bind L1 completes.
     * dt_bias is F32 on disk, directly pointed -- no widen area involved. */
    int dt_ok_1 = check_marker((const unsigned char *)b1.kda.dt_bias, 1, dt_ti, 32);
    ck(dt_ok_1, "two-slot: bind L1 content correct", "");
    int dt_ok_0b = check_marker((const unsigned char *)b0.kda.dt_bias, 0, dt_ti, 32);
    ck(dt_ok_0b, "two-slot: L0 still intact after L1 bind", "");

    /* Verify both slots occupied with distinct layers */
    int a = tr.layer_of[0], b = tr.layer_of[1];
    ck(a >= 0 && b >= 0, "two-slot: both ring slots occupied", "");
    ck(a != b, "two-slot: slots hold different layers", "");
    ck((a == 0 && b == 1) || (a == 1 && b == 0),
       "two-slot: layers 0 and 1 in ring", "");

    /* ---- ring wrap-around: L2 through 2 slots ---- */
    /* L2 evicts the oldest slot (which holds L0 since L1 was loaded second).
     * Prefetch L2 → claims slot 0, evicts L0. */
    k3_trunk_prefetch(&tr, 2);

    /* Bind L2 -- blocks on reader, publishes slot 0 */
    K3LayerBind b2; memset(&b2, 0, sizeof b2);
    if (k3_trunk_bind(&tr, c, 2, &b2) != 0) { k3_trunk_close(&tr); return 1; }

    /* After L2 bind: slot 0 should hold L2 (evicted L0), slot 1 still holds L1 */
    ck(tr.layer_of[0] == 2, "ring-wrap: slot 0 evicted L0, now holds L2", "");
    ck(tr.layer_of[1] == 1, "ring-wrap: slot 1 still holds L1", "");
    int slot2 = tr.slot_of[2];
    ck(slot2 == 0, "ring-wrap: L2 landed in recycled slot 0", "");

    /* Verify b1 pointers (slot 1) still have L1 content */
    int dt_ok_1c = check_marker((const unsigned char *)b1.kda.dt_bias, 1, dt_ti, 32);
    ck(dt_ok_1c, "ring-wrap: L1 survives L2 eviction of L0", "");

    /* Now rebind L0 -- evicts slot 1 (L1), loads L0 into slot 1 */
    k3_trunk_prefetch(&tr, 0);
    if (k3_trunk_bind(&tr, c, 0, &b0) != 0) { k3_trunk_close(&tr); return 1; }

    ck(tr.layer_of[1] == 0, "ring-wrap: rebind L0 evicted L1 from slot 1", "");
    int dt_ok_0c = check_marker((const unsigned char *)b0.kda.dt_bias, 0, dt_ti, 32);
    ck(dt_ok_0c, "ring-wrap: rebind L0 content correct", "");

    /* b2 pointers (slot 0) still have L2 content.
     * L2 is MLA -- use lay.in_norm (tensor index 0, always present). */
    int dt_ok_2c = check_marker((const unsigned char *)b2.lay.in_norm, 2, 0, 64);
    ck(dt_ok_2c, "ring-wrap: L2 survives L0 rebind", "");

    k3_trunk_close(&tr);
    return 0;
}

/* Truncated read: a fresh open, then truncate trunk.bin at layer 2's file_off.
 * Prefetch+bind of L2 must fail and leave no slot published. */
static int test_truncated(const char *dir, const K3Cfg *c)
{
    K3Trunk tr;
    if (k3_trunk_open(&tr, dir, c, 24576) != 0) {
        fprintf(stderr, "TRUNK OPEN FAILED (truncated)\n"); return 1;
    }

    /* Truncate the file at layer 2's file_off. In our fixture it is 8192. */
    /* First, use layer 0 and 1 to warm the slots, so the L2 prefetch is tested
     * in isolation on a dirty ring. */
    K3LayerBind b0; memset(&b0, 0, sizeof b0);
    K3LayerBind b1; memset(&b1, 0, sizeof b1);
    if (k3_trunk_bind(&tr, c, 0, &b0) != 0) { k3_trunk_close(&tr); return 1; }
    k3_trunk_prefetch(&tr, 1);
    if (k3_trunk_bind(&tr, c, 1, &b1) != 0) { k3_trunk_close(&tr); return 1; }

    /* Now truncate the file at layer 2's file_off = 8192 */
    char path[1024];
    snprintf(path, sizeof path, "%s/trunk.bin", dir);
    int wfd = open(path, O_WRONLY);
    if (wfd < 0) { perror("truncate open"); k3_trunk_close(&tr); return 1; }
    if (ftruncate(wfd, (off_t)tr.lay[2].file_off) != 0) {
        perror("ftruncate"); close(wfd); k3_trunk_close(&tr); return 1;
    }
    close(wfd);

    /* Initialize b2 with a byte sentinel, then verify it is unchanged after failure. */
    K3LayerBind b2;
    memset(&b2, 0xAB, sizeof b2);

    /* Prefetch L2 -- the failed read must not be counted as a miss */
    uint64_t misses_before = tr.misses;
    k3_trunk_prefetch(&tr, 2);

    int rc = k3_trunk_bind(&tr, c, 2, &b2);
    ck(rc == -1, "truncated: bind L2 returns -1", "");
    ck(tr.misses == misses_before,
       "truncated: misses unchanged after failed read", "");

    /* Verify no slot publishes layer 2: slot_of[2] untouched, and every
     * layer_of[*] is not 2. The failed prefetch must not corrupt the ring. */
    ck(tr.slot_of[2] == -1, "truncated: slot_of[2] still -1", "");
    {
        int bad = 0;
        for (int i = 0; i < tr.nslot; i++)
            if (tr.layer_of[i] == 2) bad++;
        ck(bad == 0, "truncated: no slot publishes layer 2", "");
    }

    /* Verify the complete b2 struct is byte-identical to the sentinel. */
    {
        unsigned char sentinel[sizeof b2];
        memset(sentinel, 0xAB, sizeof sentinel);
        int unchanged = (memcmp(&b2, sentinel, sizeof b2) == 0);
        ck(unchanged, "truncated: b2 struct unchanged by failed bind", "");
    }

    k3_trunk_close(&tr);
    return 0;
}

static int test_rows(const char *dir, const K3Cfg *c)
{
    K3Trunk tr, resident;
    if (k3_trunk_open_rows(&tr, dir, c, 32768)) return 1;
    if (k3_trunk_open(&resident, dir, c, 2 * FIXTURE_RUN_BYTES + 32768)) {
        k3_trunk_close(&tr); return 1;
    }
    /* The cap is derived from the same budget, so this holds by construction: it
     * documents the accounting rather than testing the memory bound independently. */
    ck(tr.row_buffer_bytes + tr.small_buffer_bytes < 32768,
       "rows: bounded two-buffer arena", "metadata also charged at allocation");
    /* The comparisons below can only see a misplaced tile if rows differ: see fixture_byte. */
    {
        K3LayerBind d;
        const int ok = !k3_trunk_bind(&resident, c, 0, &d) &&
                       memcmp(d.lay.dense_gate, (const unsigned char *)d.lay.dense_gate +
                              HIDDEN * 2, HIDDEN * 2) != 0;
        ck(ok, "rows: fixture rows differ", "a repeated-byte fixture hides misplaced tiles");
    }
    int same = 1;
    for (int walk = 0; walk < 2; walk++) {
        for (int L = 0; L < N_LAYERS; L++) {
            K3LayerBind a, b;
            if (k3_trunk_bind(&tr, c, L, &a) || k3_trunk_bind(&resident, c, L, &b)) {
                same = 0; break;
            }
            float x[DENSE_INTER > HIDDEN ? DENSE_INTER : HIDDEN];
            float got[DENSE_INTER > HIDDEN ? DENSE_INTER : HIDDEN];
            float want[DENSE_INTER > HIDDEN ? DENSE_INTER : HIDDEN];
            for (size_t i = 0; i < sizeof x / sizeof *x; i++) x[i] = (float)((int)i % 7 - 3) / 8;
            if (L == 0) {
                k3_mmw(got, x, a.lay.dense_gate, a.lay.wdt, HIDDEN, DENSE_INTER);
                k3_mmw(want, x, b.lay.dense_gate, b.lay.wdt, HIDDEN, DENSE_INTER);
                if (memcmp(got, want, sizeof(float) * DENSE_INTER)) same = 0;
                k3_mmw(got, x, a.lay.dense_down, a.lay.wdt, DENSE_INTER, HIDDEN);
                k3_mmw(want, x, b.lay.dense_down, b.lay.wdt, DENSE_INTER, HIDDEN);
                if (memcmp(got, want, sizeof(float) * HIDDEN)) same = 0;
            } else {
                k3_mmw(got, x, a.moe.up, a.moe.wdt, LATENT, HIDDEN);
                k3_mmw(want, x, b.moe.up, b.moe.wdt, LATENT, HIDDEN);
                if (memcmp(got, want, sizeof(float) * HIDDEN)) same = 0;
            }
            if (memcmp(a.lay.in_norm, b.lay.in_norm, HIDDEN * sizeof(float))) same = 0;
            if (tr.read_error) same = 0;
            k3_trunk_prefetch(&tr, (L + 1) % N_LAYERS); /* must not race row reads */
        }
    }
    /* Layer 0's 257-row matrices span several tiles; layers 1..92 compare one-tile
     * matrices and the norms. The 92 -> 0 boundary shares the row pipeline's path with
     * every other layer boundary, since rows_run drains its reads before returning. */
    ck(same, "rows: two full walks, exact matrices", "including final -> first layer and ragged row tiles");

    /* BATCHED positions read each row tile ONCE. Three positions through apply_batch must
     * equal the resident matrix bit for bit and cost exactly what one position costs: one
     * matrix call and one pass of bytes. Before apply_batch existed the batch fell back to
     * one pass per position, which is the reread this checks is gone. */
    {
        K3LayerBind a, b;
        int ok = !k3_trunk_bind(&tr, c, 0, &a) && !k3_trunk_bind(&resident, c, 0, &b);
        enum { TB = 3, W = DENSE_INTER > HIDDEN ? DENSE_INTER : HIDDEN };
        float xb[TB * W], got[TB * W], want[TB * W];
        for (int i = 0; i < TB * W; i++) xb[i] = (float)((i * 7) % 11 - 5) / 4;
        if (ok) {
            const uint64_t calls0 = tr.matrix_calls, bytes0 = tr.bytes_read;
            k3_mmw(got, xb, a.lay.dense_gate, a.lay.wdt, HIDDEN, DENSE_INTER);
            const uint64_t one_calls = tr.matrix_calls - calls0, one_bytes = tr.bytes_read - bytes0;
            const uint64_t calls1 = tr.matrix_calls, bytes1 = tr.bytes_read;
            k3_mmw_batch(got, xb, a.lay.dense_gate, a.lay.wdt, HIDDEN, DENSE_INTER, TB);
            k3_mmw_batch(want, xb, b.lay.dense_gate, b.lay.wdt, HIDDEN, DENSE_INTER, TB);
            ok = !memcmp(got, want, sizeof(float) * TB * DENSE_INTER) &&
                 tr.matrix_calls - calls1 == one_calls && one_calls == 1 &&
                 tr.bytes_read - bytes1 == one_bytes && one_bytes > 0 && !tr.read_error;
            /* and the other orientation, many rows short, into a strided output */
            k3_mmw_batch_ld(got, W, xb, W, a.lay.dense_down, a.lay.wdt, DENSE_INTER, HIDDEN, TB);
            k3_mmw_batch_ld(want, W, xb, W, b.lay.dense_down, b.lay.wdt, DENSE_INTER, HIDDEN, TB);
            for (int t = 0; t < TB; t++)
                if (memcmp(got + t * W, want + t * W, sizeof(float) * HIDDEN)) ok = 0;
        }
        ck(ok, "rows: batched positions, one pass", "3 positions == resident bitwise, bytes == 1 position");
    }
    k3_trunk_close(&resident);
    K3LayerBind b;
    if (k3_trunk_bind(&tr, c, 0, &b)) { k3_trunk_close(&tr); return 1; }
    char path[1024]; snprintf(path, sizeof path, "%s/trunk.bin", dir);
    int fd = open(path, O_WRONLY);
    if (fd < 0) { k3_trunk_close(&tr); return 1; }
    const int rc = ftruncate(fd, 0); close(fd);
    float x[HIDDEN] = {0}, y[DENSE_INTER];
    k3_mmw(y, x, b.lay.dense_gate, b.lay.wdt, HIDDEN, DENSE_INTER);
    ck(rc == 0 && tr.read_error, "rows: failed read is sticky", "no partial layer may be emitted");
    ck(k3_trunk_bind(&tr, c, 1, &b) != 0, "rows: failure refuses next layer", "");
    k3_trunk_close(&tr);
    ck(k3_trunk_open_rows(&tr, dir, c, 1) != 0, "rows: insufficient budget refused", "");
    return 0;
}

int main(void)
{
    /* Construct a minimal K3Cfg matching the fixture dimensions. */
    K3Cfg c; memset(&c, 0, sizeof c);
    c.hidden       = HIDDEN;
    c.n_layers     = N_LAYERS;
    c.vocab        = 1000;
    c.rms_eps      = 1e-5f;
    c.kda_heads    = KDA_HEADS;
    c.kda_head_dim = KDA_HEAD_DIM;
    c.conv_k       = CONV_K;
    c.gate_lb      = -5.0f;
    c.n_heads      = 2;
    c.q_lora       = 4;
    c.kv_lora      = 4;
    c.qk_nope      = 4;
    c.qk_rope      = 2;
    c.v_head       = 4;
    c.mla_out_gate = 0;       /* skip g_proj for smaller fixture */
    c.n_experts    = N_EXPERTS;
    c.topk         = 1;
    c.n_shared     = N_SHARED;
    c.latent       = LATENT;
    c.moe_inter    = MOE_INTER;
    c.routed_scale = 1.0f;
    c.moe_renorm   = 1;
    c.latent_norm  = 1;
    c.first_dense  = 1;       /* layer 0 is dense */
    c.dense_inter  = DENSE_INTER;
    c.attn_res_block = 1;
    c.situ_b1      = 4.0f;
    c.situ_b2      = 25.0f;
    /* Layer map: one-based MLA indices. 3 means layer 2 is MLA. */
    static int fa[128];
    c.n_full_attn = N_LAYERS / 3;
    for (int i = 0; i < c.n_full_attn; i++) fa[i] = (i + 1) * 3;
    c.full_attn    = fa;

    /* Create a temp directory for the fixture. mkdtemp() and /tmp are both POSIX
     * assumptions: MinGW's native runtime provides neither the function nor the
     * path, since a compiled .exe sees literal Windows paths, not the MSYS2 shell's
     * translated view of them. */
    char tmpdir[512];
#if defined(_WIN32)
    {
        char base[MAX_PATH];
        DWORD blen = GetTempPathA(sizeof base, base);
        if (blen == 0 || blen >= sizeof base) {
            fprintf(stderr, "GetTempPathA failed\n"); return 2;
        }
        int made = 0;
        for (unsigned attempt = 0; attempt < 1000; attempt++) {
            snprintf(tmpdir, sizeof tmpdir, "%sk3_test_trunk_%lu_%u",
                     base, (unsigned long)GetCurrentProcessId(), attempt);
            if (CreateDirectoryA(tmpdir, NULL)) { made = 1; break; }
        }
        if (!made) { fprintf(stderr, "CreateDirectoryA failed\n"); return 2; }
    }
#else
    snprintf(tmpdir, sizeof tmpdir, "/tmp/k3_test_trunk_XXXXXX");
    if (!mkdtemp(tmpdir)) {
        perror("mkdtemp"); return 2;
    }
#endif

    if (gen_fixture(tmpdir) != 0) {
        fprintf(stderr, "FAILED to generate fixture in %s\n", tmpdir);
        return 2;
    }

    printf("trunk streaming regression\n"
           "  fixture: %s (%d layers, %.0f KB)\n\n", tmpdir, N_LAYERS,
           (double)N_LAYERS * FIXTURE_RUN_BYTES / 1024.0);

    /* §1 one-slot budget */
    if (N_LAYERS == 3 && test_one_slot(tmpdir, &c) != 0) g_fail++;
    printf("\n");

    /* §2 two-slot budget (isolation + ring-wrap) */
    if (N_LAYERS == 3 && test_two_slot(tmpdir, &c) != 0) g_fail++;
    printf("\n");

    /* §3 truncated read (fresh open) */
    /* Re-generate the fixture since test_two_slot closed and the file is still intact. */
    if (gen_fixture(tmpdir) != 0) {
        fprintf(stderr, "FAILED to re-generate fixture\n"); g_fail++;
    } else {
        if (N_LAYERS == 3 && test_truncated(tmpdir, &c) != 0) g_fail++;
    }
    if (gen_fixture(tmpdir) || test_rows(tmpdir, &c)) g_fail++;

    /* Clean up temp files */
    {
        char p[1024];
        snprintf(p, sizeof p, "%s/trunk.bin", tmpdir);  remove(p);
        snprintf(p, sizeof p, "%s/trunk.json", tmpdir); remove(p);
        rmdir(tmpdir);
    }

    printf("\n%s\n", g_fail ? "TRUNK TESTS FAILED" : "TRUNK TESTS PASSED");
    return g_fail ? 1 : 0;
}
