#include "models.h"

// fixed by the architecture; only the input (n_embd) and output (n_cls_out) vary
static const int64_t TRAJ_FC_DIM[5] = { 4096, 1024, 512, 256, 128 };

void llama_model_robottrack::load_arch_hparams(llama_model_loader & ml) {
    llama_model_minicpm::load_arch_hparams(ml);

    // the wrapper feeds inputs_embeds, so MiniCPM's scale_emb never applies - default
    // to 1.0 instead of inheriting 12.0 when a checkpoint is missing the key
    hparams.f_embedding_scale = 1.0f;
    ml.get_key(LLM_KV_EMBEDDING_SCALE, hparams.f_embedding_scale, /*required=*/false);

    ml.get_key(LLM_KV_ROBOTTRACK_NUM_WAYPOINTS,  n_traj_waypoints,  /*required=*/false);
    ml.get_key(LLM_KV_ROBOTTRACK_ACTION_DIM,     n_traj_action_dim, /*required=*/false);

    if (hparams.n_cls_out != n_traj_waypoints * n_traj_action_dim) {
        throw std::runtime_error(format("%s: classifier.output_labels gives %u outputs, "
                    "but num_waypoints*action_dim = %u", __func__,
                    hparams.n_cls_out, n_traj_waypoints * n_traj_action_dim));
    }
}

void llama_model_robottrack::load_arch_tensors(llama_model_loader & ml) {
    llama_model_minicpm::load_arch_tensors(ml);

    LLAMA_LOAD_LOCALS;

    const int64_t n_out = hparams.n_cls_out;

    // traj.{time,stream,camera}_embd / traj.control_query never enter the graph - the caller adds
    // those rows to the input sequence itself, reading them straight out of the .gguf. They are
    // still claimed here (the loader requires every tensor in the file to be accounted for), but
    // LLM_TENSOR_INFOS maps them to GGML_OP_NONE so no model buffer is allocated for them.
    create_tensor(tn(LLM_TENSOR_TRAJ_TIME_EMBD,   "weight"), {}, TENSOR_NOT_REQUIRED);
    create_tensor(tn(LLM_TENSOR_TRAJ_STREAM_EMBD, "weight"), {}, TENSOR_NOT_REQUIRED);
    create_tensor(tn(LLM_TENSOR_TRAJ_CAMERA_EMBD, "weight"), {}, TENSOR_NOT_REQUIRED);
    create_tensor(tn(LLM_TENSOR_TRAJ_CONTROL,     "weight"), {}, TENSOR_NOT_REQUIRED);

    traj_norm_in_w = create_tensor(tn(LLM_TENSOR_TRAJ_NORM_IN, "weight"), {n_embd}, 0);
    traj_norm_in_b = create_tensor(tn(LLM_TENSOR_TRAJ_NORM_IN, "bias"),   {n_embd}, 0);

    const llm_tensor fc[5] = {
        LLM_TENSOR_TRAJ_FC1, LLM_TENSOR_TRAJ_FC2, LLM_TENSOR_TRAJ_FC3,
        LLM_TENSOR_TRAJ_FC4, LLM_TENSOR_TRAJ_FC5,
    };
    int64_t n_in = n_embd;
    for (int i = 0; i < 5; ++i) {
        traj_fc_w[i] = create_tensor(tn(fc[i], "weight"), {n_in, TRAJ_FC_DIM[i]}, 0);
        traj_fc_b[i] = create_tensor(tn(fc[i], "bias"),   {TRAJ_FC_DIM[i]}, 0);
        n_in = TRAJ_FC_DIM[i];
    }

    traj_norm_out_w = create_tensor(tn(LLM_TENSOR_TRAJ_NORM_OUT, "weight"), {n_in}, 0);
    traj_norm_out_b = create_tensor(tn(LLM_TENSOR_TRAJ_NORM_OUT, "bias"),   {n_in}, 0);

    traj_out_w = create_tensor(tn(LLM_TENSOR_TRAJ_OUT, "weight"), {n_in, n_out}, 0);
    traj_out_b = create_tensor(tn(LLM_TENSOR_TRAJ_OUT, "bias"),   {n_out}, 0);

    traj_out_scale = create_tensor(tn(LLM_TENSOR_TRAJ_OUT_SCALE, "weight"), {n_traj_action_dim}, 0);
}

void llama_model_robottrack::build_arch_head(llm_graph_context * llm) const {
    ggml_tensor * cur = llm->res->t_embd_pooled;
    if (!cur) {
        return; // embeddings disabled, nothing pooled to feed the head
    }

    ggml_context * ctx0 = llm->ctx0;

    auto norm = [&](ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
        x = ggml_norm(ctx0, x, 1e-5f);
        return ggml_add(ctx0, ggml_mul(ctx0, x, w), b);
    };

    cur = norm(cur, traj_norm_in_w, traj_norm_in_b);
    for (int i = 0; i < 5; ++i) {
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, traj_fc_w[i], cur), traj_fc_b[i]);
        cur = ggml_gelu_erf(ctx0, cur); // nn.GELU() is the exact erf form, not the tanh approximation
    }
    cur = norm(cur, traj_norm_out_w, traj_norm_out_b);
    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, traj_out_w, cur), traj_out_b);
    cur = ggml_tanh(ctx0, cur);

    // output_scale is per action dim, so view the waypoints as [action_dim, n_waypoints]
    // to broadcast it, then fold back to a flat row
    const int64_t n_seqs = cur->ne[1];
    cur = ggml_reshape_3d(ctx0, cur, n_traj_action_dim, n_traj_waypoints, n_seqs);
    cur = ggml_mul(ctx0, cur, traj_out_scale);
    cur = ggml_reshape_2d(ctx0, cur, (int64_t) hparams.n_cls_out, n_seqs);

    llm->cb(cur, "result_embd_pooled", -1);
    llm->res->t_embd_pooled = cur;

    ggml_build_forward_expand(llm->gf, cur);
}
