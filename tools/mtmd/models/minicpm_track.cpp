#include "models.h"

// MiniCPM-RobotTrack dual vision tower: DINOv3 + SigLIP encoded in one graph,
// channel-concatenated into a 1536-dim fused grid (ref: vllm DualVisionTower).
// Both encoders consume the same /255 image; per-tower normalization, the DINOv3
// axial RoPE cos/sin, and the SigLIP 27x27->24x24 adaptive-avg-pool matrix are
// baked into the mmproj (they depend only on the fixed 384px input).

// DINOv3 axial RoPE on patch tokens; cur is [d_head, n_head, n_pos, B].
// cos/sin are [d_head, n_pos] with cls+register rows pre-set to identity.
ggml_tensor * clip_graph_minicpm_track::apply_dino_rope(ggml_tensor * cur) {
    const int64_t d_head = cur->ne[0];
    const int64_t n_head = cur->ne[1];
    const int64_t n_pos  = cur->ne[2];
    const int64_t B      = cur->ne[3];

    ggml_tensor * cos = ggml_reshape_4d(ctx0, model.minicpm_track_dino_rope_cos, d_head, 1, n_pos, 1);
    ggml_tensor * sin = ggml_reshape_4d(ctx0, model.minicpm_track_dino_rope_sin, d_head, 1, n_pos, 1);

    // rotate_half(x) = concat(-x[d/2:], x[:d/2]) along dim 0
    ggml_tensor * x1 = ggml_cont(ctx0, ggml_view_4d(ctx0, cur, d_head/2, n_head, n_pos, B,
                            cur->nb[1], cur->nb[2], cur->nb[3], 0));
    ggml_tensor * x2 = ggml_cont(ctx0, ggml_view_4d(ctx0, cur, d_head/2, n_head, n_pos, B,
                            cur->nb[1], cur->nb[2], cur->nb[3], (d_head/2) * ggml_element_size(cur)));
    ggml_tensor * rot = ggml_concat(ctx0, ggml_neg(ctx0, x2), x1, 0);

    return ggml_add(ctx0, ggml_mul(ctx0, cur, cos), ggml_mul(ctx0, rot, sin));
}

// Generic ViT tower over its own layer vector (mirrors clip_graph::build_vit,
// but reads per-tower tensors so both towers live in one graph).
ggml_tensor * clip_graph_minicpm_track::build_tower(
        ggml_tensor * inp,
        const std::vector<clip_layer> & layers,
        int n_head_,
        float attn_scale,
        float tower_eps,
        ffn_op_type ffn_t,
        ggml_tensor * post_ln_w,
        ggml_tensor * post_ln_b,
        bool use_dino_rope) {
    const int64_t n_embd_ = inp->ne[0];
    const int64_t n_pos   = inp->ne[1];
    const int64_t d_head_ = n_embd_ / n_head_;

    ggml_tensor * inpL = inp;
    for (int il = 0; il < (int) layers.size(); ++il) {
        const auto & layer = layers[il];

        ggml_tensor * cur = build_norm(inpL, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, tower_eps, il);

        // self-attention
        ggml_tensor * Q = build_mm(layer.q_w, cur);
        if (layer.q_b) { Q = ggml_add(ctx0, Q, layer.q_b); }
        ggml_tensor * K = build_mm(layer.k_w, cur);
        if (layer.k_b) { K = ggml_add(ctx0, K, layer.k_b); }
        ggml_tensor * V = build_mm(layer.v_w, cur);
        if (layer.v_b) { V = ggml_add(ctx0, V, layer.v_b); }

        Q = ggml_reshape_4d(ctx0, Q, d_head_, n_head_, n_pos, 1);
        K = ggml_reshape_4d(ctx0, K, d_head_, n_head_, n_pos, 1);
        V = ggml_reshape_4d(ctx0, V, d_head_, n_head_, n_pos, 1);

        if (use_dino_rope) {
            Q = apply_dino_rope(Q);
            K = apply_dino_rope(K);
        }

        cur = build_attn(layer.o_w, layer.o_b, Q, K, V, nullptr, attn_scale, il);

        if (layer.ls_1_w) { cur = ggml_mul(ctx0, cur, layer.ls_1_w); }
        cur  = ggml_add(ctx0, cur, inpL);
        inpL = cur;

        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, tower_eps, il);
        cur = build_ffn(cur,
            layer.ff_up_w, layer.ff_up_b,
            nullptr, nullptr,
            layer.ff_down_w, layer.ff_down_b,
            ffn_t, il);
        if (layer.ls_2_w) { cur = ggml_mul(ctx0, cur, layer.ls_2_w); }

        cur  = ggml_add(ctx0, inpL, cur);
        inpL = cur;
    }

    if (post_ln_w) {
        inpL = build_norm(inpL, post_ln_w, post_ln_b, NORM_TYPE_NORMAL, tower_eps, -1);
    }
    return inpL;
}

// Uniform adaptive-avg-pool of a [C, grid*grid] fused map down to [C, out*out].
// grid is divisible by out (24->8 window 3, 24->2 window 12), so torch
// adaptive_avg_pool2d degenerates to plain non-overlapping average pooling.
// Token order (w + grid*h) is preserved to match the torch reshape/flatten.
ggml_tensor * clip_graph_minicpm_track::pool_fused(ggml_tensor * fused, int grid, int out) {
    const int64_t C = fused->ne[0];
    const int     k = grid / out;
    ggml_tensor * g = ggml_reshape_4d(ctx0, fused, C, grid, grid, 1);      // [C, W, H, 1]
    g = ggml_cont(ctx0, ggml_permute(ctx0, g, 2, 0, 1, 3));                // [W, H, C, 1]
    g = ggml_pool_2d(ctx0, g, GGML_OP_POOL_AVG, k, k, k, k, 0, 0);         // [out, out, C, 1]
    g = ggml_cont(ctx0, ggml_permute(ctx0, g, 1, 2, 0, 3));                // [C, out, out, 1]
    g = ggml_reshape_2d(ctx0, g, C, (int64_t) out * out);                  // [C, out*out]
    return g;
}

ggml_cgraph * clip_graph_minicpm_track::build() {
    ggml_tensor * inp_raw = build_inp_raw(); // [W, H, 3, 1], values in [0,1]

    // per-tower normalization in-graph (single /255 image in)
    ggml_tensor * dino_mean   = ggml_reshape_3d(ctx0, model.minicpm_track_dino_mean,     1, 1, 3);
    ggml_tensor * dino_istd   = ggml_reshape_3d(ctx0, model.minicpm_track_dino_inv_std,  1, 1, 3);
    ggml_tensor * siglip_mean = ggml_reshape_3d(ctx0, model.minicpm_track_siglip_mean,   1, 1, 3);
    ggml_tensor * siglip_istd = ggml_reshape_3d(ctx0, model.minicpm_track_siglip_inv_std,1, 1, 3);

    ggml_tensor * dino_in   = ggml_mul(ctx0, ggml_sub(ctx0, inp_raw, dino_mean),   dino_istd);
    ggml_tensor * siglip_in = ggml_mul(ctx0, ggml_sub(ctx0, inp_raw, siglip_mean), siglip_istd);

    // ---- DINOv3 tower ----
    const int d_patch = hparams.patch_size; // 16
    ggml_tensor * dino_tokens;
    {
        ggml_tensor * cur = ggml_conv_2d(ctx0, model.minicpm_track_dino_patch_w, dino_in, d_patch, d_patch, 0, 0, 1, 1);
        const int64_t np = cur->ne[0] * cur->ne[1]; // 24*24 = 576
        cur = ggml_reshape_3d(ctx0, cur, np, n_embd, 1);
        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur)); // [n_embd, np, 1]
        cur = ggml_add(ctx0, cur, model.minicpm_track_dino_patch_b);

        // prepend [cls, register(4)] -> [n_embd, 5 + np, 1]
        ggml_tensor * cls = ggml_reshape_3d(ctx0, model.minicpm_track_dino_cls, n_embd, 1, 1);
        ggml_tensor * reg = ggml_reshape_3d(ctx0, model.minicpm_track_dino_register, n_embd, model.minicpm_track_dino_register->ne[1], 1);
        cur = ggml_concat(ctx0, ggml_concat(ctx0, cls, reg, 1), cur, 1);

        const int   n_head_d = hparams.n_head;               // 6
        const float scale_d  = 1.0f / sqrtf((float)(n_embd / n_head_d)); // 1/sqrt(64)
        cur = build_tower(cur, model.minicpm_track_dino_layers, n_head_d, scale_d, hparams.eps,
                          FFN_GELU_ERF, model.minicpm_track_dino_post_ln_w, model.minicpm_track_dino_post_ln_b, true);

        // drop cls + register (first 5 tokens)
        const int n_prefix = 1 + (int) model.minicpm_track_dino_register->ne[1];
        dino_tokens = ggml_cont(ctx0, ggml_view_3d(ctx0, cur, n_embd, cur->ne[1] - n_prefix, 1,
                            cur->nb[1], cur->nb[2], (size_t) n_prefix * cur->nb[1]));
    }

    // ---- SigLIP tower ----
    const int s_embd  = hparams.siglip_n_embd;   // 1152
    const int s_patch = hparams.siglip_patch_size; // 14
    ggml_tensor * siglip_pooled;
    {
        ggml_tensor * cur = ggml_conv_2d(ctx0, model.minicpm_track_siglip_patch_w, siglip_in, s_patch, s_patch, 0, 0, 1, 1);
        const int64_t np = cur->ne[0] * cur->ne[1]; // 27*27 = 729
        cur = ggml_reshape_3d(ctx0, cur, np, s_embd, 1);
        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur)); // [s_embd, np, 1]
        cur = ggml_add(ctx0, cur, model.minicpm_track_siglip_patch_b);
        cur = ggml_add(ctx0, cur, model.minicpm_track_siglip_pos);    // learned position embedding

        const int   n_head_s = hparams.siglip_n_head;        // 16
        const float scale_s  = 1.0f / sqrtf((float)(s_embd / n_head_s)); // 1/sqrt(72)
        // SigLIP uses gelu_pytorch_tanh -> FFN_GELU (ggml_gelu)
        cur = build_tower(cur, model.minicpm_track_siglip_layers, n_head_s, scale_s, hparams.siglip_eps,
                          FFN_GELU, model.minicpm_track_siglip_post_ln_w, model.minicpm_track_siglip_post_ln_b, false);

        // adaptive_avg_pool2d 27x27 -> 24x24 as a baked [n_in, n_out] matmul
        ggml_tensor * st = ggml_cont(ctx0, ggml_transpose(ctx0, cur));   // [np, s_embd]
        st = ggml_mul_mat(ctx0, model.minicpm_track_siglip_pool, st);               // [576, s_embd]
        siglip_pooled = ggml_cont(ctx0, ggml_transpose(ctx0, st));       // [s_embd, 576]
    }

    // ---- fuse: channel-concat [dino, siglip] ----
    ggml_tensor * fused = ggml_concat(ctx0, dino_tokens, siglip_pooled, 0); // [n_embd + s_embd, 576]
    fused = ggml_reshape_2d(ctx0, fused, fused->ne[0], fused->ne[1]);
    cb(fused, "minicpm_track_fused", -1);

    // ---- coarse (2x2) + fine (8x8) adaptive-avg pools of the 24x24 grid ----
    const int grid = (int) (sqrtf((float) fused->ne[1]) + 0.5f); // 24
    ggml_tensor * coarse = pool_fused(fused, grid, 2); // [1536, 4]
    ggml_tensor * fine   = pool_fused(fused, grid, 8); // [1536, 64]

    // output layout: coarse (4) first, then fine (64) -> 68 tokens
    ggml_tensor * pooled = ggml_concat(ctx0, coarse, fine, 1); // [1536, 68]

    // ---- VisionProjector: LayerNorm -> Linear -> GELU(erf) -> Linear -> [1024, 68] ----
    ggml_tensor * proj = build_norm(pooled, model.minicpm_track_mm_norm_w, model.minicpm_track_mm_norm_b,
                                    NORM_TYPE_NORMAL, /*norm_eps=*/1e-5f, -1);
    proj = ggml_add(ctx0, build_mm(model.minicpm_track_mm_fc1_w, proj), model.minicpm_track_mm_fc1_b);
    proj = ggml_gelu_erf(ctx0, proj);
    proj = ggml_add(ctx0, build_mm(model.minicpm_track_mm_fc2_w, proj), model.minicpm_track_mm_fc2_b);
    cb(proj, "minicpm_track_proj", -1);

    ggml_build_forward_expand(gf, proj);
    return gf;
}
