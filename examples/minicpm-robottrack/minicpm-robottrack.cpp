// End-to-end MiniCPM-RobotTrack driver: the whole model runs in llama.cpp.
//
// Per frame, a rolling 32-frame window goes through:
//
//   frame input     : directory of *.jpg/*.png, a single image, or a video file
//   mmproj (mtmd)   : frame -> coarse[4,1024] + fine[64,1024]
//   window assembly : fine = last frame; history = coarse of the earlier frames,
//                     left-padded to 31 by repeating the oldest frame
//   sequence        : text_emb(unscaled) ++ [marker(i,0);coarse]x31
//                     ++ [marker(31,1);fine] ++ control_query  -> seq[S,1024]
//   model (llama)   : seq goes in via batch.embd; LLM_ARCH_ROBOTTRACK pools the
//                     trailing control query and applies the funnel head in-graph,
//                     so llama_get_embeddings_seq() returns the scaled [8,3]
//
// The markers and the control query are consumed while assembling the input, i.e.
// outside the graph, so the driver reads them straight out of the GGUF. Text
// embeddings are read raw as well: the RobotTrack wrapper feeds inputs_embeds, so
// MiniCPM's scale_emb must not be applied.
//
// Media handling is entirely public libmtmd, so this file only needs mtmd.h and
// mtmd-helper.h - no mtmd-internal headers.

#include "mtmd.h"
#include "mtmd-helper.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static const int WINDOW_FRAMES    = 32; // 31 history + 1 current
static const int HISTORY_FRAMES   = 31;
static const int COARSE_PER_FRAME = 4;
static const int FINE_TOKENS      = 64;
static const int VISION_TOKENS    = COARSE_PER_FRAME + FINE_TOKENS; // 68
static const int NUM_WAYPOINTS    = 8;
static const int ACTION_DIM       = 3;

// input rows that do not change between frames
struct prompt_parts {
    int                n_embd  = 0;
    int                n_text  = 0;
    std::vector<float> text;    // [n_text,        n_embd] unscaled token embedding rows
    std::vector<float> markers; // [WINDOW_FRAMES, n_embd] one marker row per window slot
    std::vector<float> control; // [n_embd]
};

static const float * f32_tensor(ggml_context * ctx, const char * name) {
    ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t) {
        fprintf(stderr, "missing tensor %s in model\n", name);
        exit(1);
    }
    return (const float *) t->data;
}

// llama.cpp exposes no raw tensor data, so reopen the file to get at these rows
static bool load_prompt_parts(const char * model_path, const llama_vocab * vocab,
                              const char * instruction, int n_embd, prompt_parts & out) {
    ggml_context * ctx = nullptr;
    gguf_init_params gp = { /*no_alloc=*/false, /*ctx=*/&ctx };
    gguf_context * gg = gguf_init_from_file(model_path, gp);
    if (!gg || !ctx) {
        fprintf(stderr, "cannot reopen %s as gguf\n", model_path);
        return false;
    }
    auto fail = [&](const char * msg) {
        fprintf(stderr, "%s\n", msg);
        gguf_free(gg);
        ggml_free(ctx);
        return false;
    };

    out.n_embd = n_embd;

    std::vector<llama_token> toks(256);
    const int T = llama_tokenize(vocab, instruction, (int) strlen(instruction), toks.data(),
                                 (int) toks.size(), /*add_special=*/true, /*parse_special=*/false);
    if (T < 0) {
        return fail("instruction is longer than the tokenizer scratch buffer");
    }
    printf("[text] T=%d ids=", T);
    for (int i = 0; i < T; ++i) {
        printf("%d ", toks[i]);
    }
    printf("\n");

    ggml_tensor * tok_embd = ggml_get_tensor(ctx, "token_embd.weight");
    if (!tok_embd || (int) tok_embd->ne[0] != n_embd) {
        return fail("token_embd.weight is missing or has the wrong shape");
    }
    if (tok_embd->type != GGML_TYPE_F16) {
        return fail("token_embd.weight is not f16 (convert the model with --outtype f16)");
    }
    const ggml_fp16_t * rows = (const ggml_fp16_t *) tok_embd->data;
    out.n_text = T;
    out.text.resize((size_t) T * n_embd);
    for (int i = 0; i < T; ++i) {
        ggml_fp16_to_fp32_row(rows + (size_t) toks[i] * n_embd,
                              out.text.data() + (size_t) i * n_embd, n_embd);
    }

    // marker row = time[t] + stream[s] + camera[0]
    const float * m_time   = f32_tensor(ctx, "traj.time_embd.weight");
    const float * m_stream = f32_tensor(ctx, "traj.stream_embd.weight");
    const float * m_camera = f32_tensor(ctx, "traj.camera_embd.weight");

    out.markers.resize((size_t) WINDOW_FRAMES * n_embd);
    for (int t = 0; t < WINDOW_FRAMES; ++t) {
        const int s = (t == HISTORY_FRAMES) ? 1 : 0; // history stream 0, current stream 1
        for (int j = 0; j < n_embd; ++j) {
            out.markers[(size_t) t * n_embd + j] = m_time  [(size_t) t * n_embd + j]
                                                 + m_stream[(size_t) s * n_embd + j]
                                                 + m_camera[j];
        }
    }

    const float * control = f32_tensor(ctx, "traj.control_query.weight");
    out.control.assign(control, control + n_embd);

    gguf_free(gg);
    ggml_free(ctx);
    return true;
}

// yields raw (un-resized) frames, oldest first, from a list of image files or a video
struct frame_source {
    mtmd_context * mctx = nullptr;

    std::vector<std::string> paths;      // image mode
    size_t                   next_path = 0;

    mtmd_helper_video * video     = nullptr; // video mode
    int                 video_idx = 0;

    int n_frames = -1; // -1 = unknown

    ~frame_source() {
        if (video) {
            mtmd_helper_video_free(video);
        }
    }

    // nullptr = end of stream; the caller owns the returned bitmap
    mtmd_bitmap * next(std::string & name) {
        return video ? next_video_frame(name) : next_image_file(name);
    }

  private:
    mtmd_bitmap * next_image_file(std::string & name) {
        while (next_path < paths.size()) {
            const std::string p = paths[next_path++];
            auto w = mtmd_helper_bitmap_init_from_file(mctx, p.c_str(), /*placeholder=*/false);
            if (!w.bitmap) {
                fprintf(stderr, "warn: cannot decode %s, skipping\n", p.c_str());
                continue;
            }
            if (w.video_ctx) { // a video file sitting in the frame directory
                mtmd_bitmap_free(w.bitmap);
                mtmd_helper_video_free(w.video_ctx);
                fprintf(stderr, "warn: %s is a video, skipping\n", p.c_str());
                continue;
            }
            name = fs::path(p).filename().string();
            return w.bitmap;
        }
        return nullptr;
    }

    mtmd_bitmap * next_video_frame(std::string & name) {
        for (;;) {
            mtmd_bitmap * bm  = nullptr;
            char        * txt = nullptr;
            const int32_t r = mtmd_helper_video_read_next(video, &bm, &txt);
            if (r == -2) {
                fprintf(stderr, "error: video read failed at frame %d\n", video_idx);
                return nullptr;
            }
            if (r != 0) {
                return nullptr; // EOF
            }
            if (txt) {
                // the helper emits a "Video:" preamble and periodic timestamps;
                // we drive our own text prompt, so drop them
                free(txt);
                continue;
            }
            if (!bm) {
                continue;
            }
            char buf[32];
            snprintf(buf, sizeof(buf), "frame_%05d.png", video_idx++);
            name = buf;
            return bm;
        }
    }
};

static std::vector<std::string> list_frame_files(const std::string & dir) {
    // matches the reference demo: sorted(*.jpg) followed by sorted(*.png)
    std::vector<std::string> jpg, png;
    for (const auto & e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) {
            continue;
        }
        std::string ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return (char) tolower(c); });
        if      (ext == ".jpg") jpg.push_back(e.path().string());
        else if (ext == ".png") png.push_back(e.path().string());
    }
    std::sort(jpg.begin(), jpg.end());
    std::sort(png.begin(), png.end());
    jpg.insert(jpg.end(), png.begin(), png.end());
    return jpg;
}

// Keep videos away from mtmd_helper_bitmap_init_from_file(): its video fallthrough
// is buffer-based, and feeding a seekable container to ffprobe over a pipe dies with
// SIGPIPE. Anything not recognized here is treated as a video and fails cleanly in
// ffprobe if it isn't one.
static bool looks_like_image(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    unsigned char m[12] = {};
    const size_t n = fread(m, 1, sizeof(m), f);
    fclose(f);
    if (n < 8) {
        return false;
    }
    const bool jpg = m[0] == 0xFF && m[1] == 0xD8 && m[2] == 0xFF;
    const bool png = memcmp(m, "\x89PNG\r\n\x1A\n", 8) == 0;
    const bool bmp = m[0] == 'B' && m[1] == 'M';
    const bool gif = memcmp(m, "GIF87a", 6) == 0 || memcmp(m, "GIF89a", 6) == 0;
    return jpg || png || bmp || gif;
}

static bool open_frame_source(frame_source & src, mtmd_context * mctx,
                             const std::string & input, float fps_target) {
    src.mctx = mctx;

    if (fs::is_directory(input)) {
        src.paths = list_frame_files(input);
        if (src.paths.empty()) {
            fprintf(stderr, "no .jpg/.png frames in %s\n", input.c_str());
            return false;
        }
        src.n_frames = (int) src.paths.size();
        printf("[input] frame directory %s (%d frames)\n", input.c_str(), src.n_frames);
        return true;
    }

    if (looks_like_image(input)) {
        src.paths.push_back(input);
        src.n_frames = 1;
        printf("[input] single image %s\n", input.c_str());
        return true;
    }

    auto vp = mtmd_helper_video_init_params_default();
    vp.fps_target            = fps_target; // <=0 -> native fps
    vp.timestamp_interval_ms = 0;          // no timestamp text chunks
    src.video = mtmd_helper_video_init(mctx, input.c_str(), vp);
    if (!src.video) {
        fprintf(stderr, "cannot open %s as image or video\n", input.c_str());
        return false;
    }
    const auto vi = mtmd_helper_video_get_info(src.video);
    src.n_frames  = vi.n_frames;
    printf("[input] video %s (%ux%u, %.2f fps, ~%d frames)\n",
           input.c_str(), vi.width, vi.height, vi.fps, vi.n_frames);
    return true;
}

// the bitmap is only kept in --reencode mode, where the whole window is
// re-encoded every step
struct win_frame {
    mtmd::bitmap_ptr   bmp;
    std::vector<float> coarse; // [COARSE_PER_FRAME, n_embd]
    std::vector<float> fine;   // [FINE_TOKENS,      n_embd]
};

// mtmd_tokenize() dispatches the MINICPM_TRACK preprocessor (bicubic stretch to 384
// + /255) and emits one image chunk of VISION_TOKENS tokens, coarse[4] then fine[64]
static bool encode_frame(mtmd_context * mctx, int n_embd, win_frame & f) {
    const mtmd_bitmap * bmps[1] = { f.bmp.get() };

    const std::string marker = mtmd_get_marker(mctx);
    mtmd_input_text txt;
    txt.text          = marker.c_str();
    txt.text_len      = marker.size();
    txt.add_special   = false;
    txt.parse_special = false;

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    if (mtmd_tokenize(mctx, chunks.ptr.get(), &txt, bmps, 1) != 0) {
        fprintf(stderr, "mtmd_tokenize failed\n");
        return false;
    }

    const mtmd_input_chunk * img = nullptr;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks.ptr.get()); ++i) {
        const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks.ptr.get(), i);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            img = c;
            break;
        }
    }
    if (!img) {
        fprintf(stderr, "mtmd_tokenize produced no image chunk\n");
        return false;
    }

    const int n_tok = (int) mtmd_input_chunk_get_n_tokens(img);
    if (n_tok != VISION_TOKENS) {
        fprintf(stderr, "unexpected vision token count %d (want %d)\n", n_tok, VISION_TOKENS);
        return false;
    }
    if (mtmd_encode_chunk(mctx, img) != 0) {
        fprintf(stderr, "mtmd_encode_chunk failed\n");
        return false;
    }
    const float * embd = mtmd_get_output_embd(mctx);
    if (!embd) {
        fprintf(stderr, "mtmd_get_output_embd returned null\n");
        return false;
    }
    // copy out now: the buffer is reused by the next encode pass
    const size_t n_coarse = (size_t) COARSE_PER_FRAME * n_embd;
    f.coarse.assign(embd, embd + n_coarse);
    f.fine.assign(embd + n_coarse, embd + (size_t) VISION_TOKENS * n_embd);
    return true;
}

// returns the number of rows written, which must equal S
static int assemble_sequence(std::vector<float> & seq, const prompt_parts & p,
                             const std::deque<win_frame> & win, const std::vector<int> & hist) {
    const int n_embd = p.n_embd;
    int r = 0;
    auto append = [&](const float * src, int n_rows) {
        memcpy(seq.data() + (size_t) r * n_embd, src, sizeof(float) * (size_t) n_rows * n_embd);
        r += n_rows;
    };

    append(p.text.data(), p.n_text);
    for (int fr = 0; fr < HISTORY_FRAMES; ++fr) {
        append(p.markers.data() + (size_t) fr * n_embd, 1);
        append(win[hist[fr]].coarse.data(), COARSE_PER_FRAME);
    }
    append(p.markers.data() + (size_t) HISTORY_FRAMES * n_embd, 1);
    append(win.back().fine.data(), FINE_TOKENS);
    append(p.control.data(), 1);
    return r;
}

static void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s <mmproj.gguf> <model.gguf> \"<instruction>\" <input> [options]\n"
        "\n"
        "  <input> is one of:\n"
        "    - a directory of *.jpg / *.png frames   (demo main path)\n"
        "    - a single image file                   (single-frame regression)\n"
        "    - a video file (mp4/mkv/...)            (needs ffmpeg/ffprobe in PATH)\n"
        "\n"
        "  options:\n"
        "    --max-frames N        stop after N frames (0 = all, default)\n"
        "    --threads N           backbone threads (default 8)\n"
        "    --vision-threads N    mmproj threads (default 4)\n"
        "    --fps F               video target fps (<=0 = native, default)\n"
        "    --control-waypoint K  waypoint used for the printed velocity (default 1)\n"
        "    --control-dt D        control timestep for the printed velocity (default 0.1)\n"
        "    --reencode            re-encode the whole window every step instead of\n"
        "                          reusing per-frame tokens (same result, ~N x slower)\n",
        argv0);
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        usage(argv[0]);
        return 1;
    }
    const char * mmproj_path   = argv[1];
    const char * model_path    = argv[2];
    const char * instruction   = argv[3];
    const std::string input    = argv[4];

    int         max_frames       = 0;
    int         n_threads        = 8;
    int         n_threads_vision = 4;
    float       fps_target       = 0.0f;
    int         control_waypoint = 1;
    float       control_dt       = 0.1f;
    bool        reencode         = false;

    for (int ai = 5; ai < argc; ++ai) {
        const std::string a = argv[ai];
        auto need = [&](const char * flag) -> const char * {
            if (ai + 1 >= argc) { fprintf(stderr, "%s needs a value\n", flag); exit(1); }
            return argv[++ai];
        };
        if      (a == "--max-frames")       max_frames       = atoi(need("--max-frames"));
        else if (a == "--threads")          n_threads        = atoi(need("--threads"));
        else if (a == "--vision-threads")   n_threads_vision = atoi(need("--vision-threads"));
        else if (a == "--fps")              fps_target       = (float) atof(need("--fps"));
        else if (a == "--control-waypoint") control_waypoint = atoi(need("--control-waypoint"));
        else if (a == "--control-dt")       control_dt       = (float) atof(need("--control-dt"));
        else if (a == "--reencode")         reencode         = true;
        else { fprintf(stderr, "unknown option: %s\n", a.c_str()); usage(argv[0]); return 1; }
    }

    if (control_waypoint < 0 || control_waypoint >= NUM_WAYPOINTS) {
        fprintf(stderr, "--control-waypoint must be in [0, %d)\n", NUM_WAYPOINTS);
        return 1;
    }

    llama_backend_init();

    llama_model * model = llama_model_load_from_file(model_path, llama_model_default_params());
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const int n_embd = llama_model_n_embd(model);

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.print_timings    = false;
    mparams.n_threads        = n_threads_vision;
    mparams.warmup           = false;
    mparams.image_min_tokens = -1;
    mparams.image_max_tokens = -1;
    mtmd::context_ptr mctx(mtmd_init_from_file(mmproj_path, model, mparams));
    if (!mctx) { fprintf(stderr, "mtmd_init_from_file failed\n"); return 1; }
    if (llama_model_n_embd_inp(model) != n_embd) {
        fprintf(stderr, "n_embd_inp %d != n_embd %d\n", llama_model_n_embd_inp(model), n_embd);
        return 1;
    }

    prompt_parts prompt;
    if (!load_prompt_parts(model_path, llama_model_get_vocab(model), instruction, n_embd, prompt)) {
        return 1;
    }

    frame_source src;
    if (!open_frame_source(src, mctx.get(), input, fps_target)) {
        return 1;
    }

    const int S = prompt.n_text + HISTORY_FRAMES * (1 + COARSE_PER_FRAME) + (1 + FINE_TOKENS) + 1;
    printf("[seq] S=%d (T=%d + %d hist + %d curr + 1 control)\n",
           S, prompt.n_text, HISTORY_FRAMES * (1 + COARSE_PER_FRAME), 1 + FINE_TOKENS);

    // the context is created once and its KV is cleared per frame
    llama_context_params lcp = llama_context_default_params();
    lcp.n_ctx = S + 8;
    lcp.n_batch = S;
    lcp.n_ubatch = S;
    // RANK pools the trailing control query and runs the trajectory head, so the
    // sequence output is n_cls_out floats rather than an n_embd embedding
    lcp.pooling_type = LLAMA_POOLING_TYPE_RANK;
    lcp.embeddings = true;
    lcp.n_threads = n_threads;
    lcp.n_threads_batch = n_threads;
    llama_context * lctx = llama_init_from_model(model, lcp);
    if (!lctx) { fprintf(stderr, "ctx init failed\n"); return 1; }

    llama_batch batch = llama_batch_init(S, n_embd, 1);
    batch.n_tokens = S;
    for (int i = 0; i < S; ++i) {
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == S - 1);
    }

    std::deque<win_frame> win;
    std::vector<float>    seq((size_t) S * n_embd);
    int                   frame_idx = 0;

    for (;; ++frame_idx) {
        if (max_frames > 0 && frame_idx >= max_frames) {
            break;
        }

        std::string name;
        mtmd::bitmap_ptr raw(src.next(name));
        if (!raw) {
            break;
        }
        win.push_back({});
        win.back().bmp = std::move(raw);
        if ((int) win.size() > WINDOW_FRAMES) {
            win.pop_front();
        }
        const int N = (int) win.size();

        // --reencode re-runs the whole window like the vLLM demo does; encoding is
        // per-frame and deterministic, so it is bit-identical to encoding just the
        // new frame, and the flag exists to check that
        for (int i = reencode ? 0 : N - 1; i < N; ++i) {
            if (!encode_frame(mctx.get(), n_embd, win[i])) {
                fprintf(stderr, "encode failed on %s (window slot %d)\n", name.c_str(), i);
                return 1;
            }
        }
        if (!reencode) {
            win.back().bmp.reset();
        }

        // history is every frame but the current one, left-padded by repeating the
        // oldest frame - always slot 0, since the window pops from the front. a
        // 1-frame window is its own history
        const int n_src = std::max(N - 1, 1);
        std::vector<int> hist(HISTORY_FRAMES, 0);
        for (int k = 0; k < n_src; ++k) {
            hist[HISTORY_FRAMES - n_src + k] = k;
        }

        const int r = assemble_sequence(seq, prompt, win, hist);
        if (r != S) { fprintf(stderr, "assembly length %d != %d\n", r, S); return 1; }

        memcpy(batch.embd, seq.data(), sizeof(float) * (size_t) S * n_embd);
        llama_memory_clear(llama_get_memory(lctx), true);
        if (llama_decode(lctx, batch) < 0) { fprintf(stderr, "decode failed\n"); return 1; }
        const float * traj = llama_get_embeddings_seq(lctx, 0);  // [24], row-major [8,3], already scaled
        if (!traj) { fprintf(stderr, "no trajectory output\n"); return 1; }

        char pos[24]; // "12/64", or just "12" when the frame count is unknown
        if (src.n_frames > 0) snprintf(pos, sizeof(pos), "%d/%d", frame_idx, src.n_frames);
        else                  snprintf(pos, sizeof(pos), "%d",    frame_idx);

        const float * v = traj + control_waypoint * ACTION_DIM;
        printf("  frame %9s N=%-2d %-22s vel=(%+.4f, %+.4f, %+.4f)\n",
               pos, N, name.c_str(), v[0] / control_dt, v[1] / control_dt, v[2] / control_dt);
        fflush(stdout);

        if (src.n_frames == 1) {
            printf("[trajectory] (x, y, yaw) * output_scale:\n");
            for (int w = 0; w < NUM_WAYPOINTS; ++w) {
                printf("  wp%-2d  % .5f  % .5f  % .5f\n", w,
                       traj[w * 3 + 0], traj[w * 3 + 1], traj[w * 3 + 2]);
            }
        }
    }

    printf("[done] %d frames\n", frame_idx);

    win.clear(); // bitmaps must go before the mtmd context they were made with
    llama_batch_free(batch);
    llama_free(lctx);
    mctx.reset();
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
