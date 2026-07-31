# MiniCPM-RobotTrack

End-to-end example for MiniCPM-RobotTrack, a vision-language-action (VLA) model that
predicts a short driving trajectory from a rolling window of camera frames plus a text
instruction.

One forward pass consumes 32 frames and returns 8 waypoints of `(x, y, yaw)`. There is
no token generation: the model is not autoregressive, and the example never calls
`llama_decode` more than once per frame.

## Model structure

Two GGUF files, matching llama.cpp's usual split between the multimodal encoder and the
language model.

```
  one frame, any resolution
            |
   bicubic stretch to 384x384, /255      (no letterbox: aspect ratio is ignored)
            |
      +-----+------------------------------+
      |                                    |
  DINOv3 ViT-S/16                    SigLIP so400m/14
  width 384, 12 layers               width 1152, 27 layers
  24x24 = 576 patches                27x27 = 729 patches
  + 1 cls + 4 register               (axial RoPE on patches only)
      |                                    |
      |                          avg pool 27x27 -> 24x24
      |                                    |
      +-----+------------------------------+
            |
      concat on width -> [576, 1536]
            |
      +-----+---------------+
      |                     |
 grid pool 24->2       grid pool 24->8
 coarse [4, 1536]      fine [64, 1536]
      |                     |
      +-----+---------------+
            |
  VisionProjector: LN -> Linear -> GELU -> Linear,  1536 -> 1024
            |
      [68, 1024]  =  coarse[4] ++ fine[64]                        <- mmproj output
```

The caller keeps a 32-frame window and assembles one sequence per frame. Markers and the
control query are read from the backbone GGUF:

```
  instruction --> token_embd, unscaled --> [T, 1024]

  window:  frame 0 .. 30  -> coarse[4] each      (history, left-padded to 31)
           frame 31       -> fine[64]            (current)

  [ text ][ mk(0,0) coarse x4 ] ... [ mk(30,0) coarse x4 ][ mk(31,1) fine x64 ][ control ]
     T    \___________________ 31 x 5 = 155 ___________________/\____ 65 ____/\____ 1 ___/

  S = T + 221 rows of 1024                        mk(t,s) = time[t] + stream[s] + camera[0]
            |
       batch.embd
            |
  MiniCPM4-0.5B: 24 layers, width 1024, 16 heads / 2 kv, causal
            |
  build_pooling()   RANK takes the row at the control position  [1024]
            |
  build_arch_head() funnel head -> trajectory [8, 3]
            |                     (extra hook, runs right after pooling)
  llama_get_embeddings_seq()
```

### mmproj (`clip.projector_type = minicpm_track`)

A dual vision tower plus a projector, run by `tools/mtmd`:

| part | shape |
|---|---|
| input | one 384x384 RGB frame, bicubic stretch (no letterbox), `/255` |
| DINOv3 tower | patch 16, 12 layers, width 384 |
| SigLIP tower | patch 14, 27 layers, width 1152 |
| fused | `[576, 1536]` per frame |
| grid pool | 24x24 -> 2x2 = `coarse[4]`, 24x24 -> 8x8 = `fine[64]` |
| VisionProjector | LayerNorm -> Linear -> GELU -> Linear |
| output | `[68, 1024]`, `coarse[4]` followed by `fine[64]` |

Per-tower mean/std normalization and the SigLIP 27x27 -> 24x24 pooling matrix are baked
into the GGUF, so the graph takes a single `/255` image.

Graph: `tools/mtmd/models/minicpm_track.cpp`. Weights live in `clip_model` under the
`minicpm_track_` prefix.

### backbone (`general.architecture = robottrack`)

MiniCPM4-0.5B plus a trajectory head:

| key | value |
|---|---|
| `block_count` | 24 |
| `embedding_length` | 1024 |
| `attention.head_count` / `_kv` | 16 / 2 |
| `feed_forward_length` | 4096 |
| `embedding_scale` | 1.0 |
| `num_waypoints` / `action_dim` | 8 / 3 |
| `classifier.output_labels` | 24 entries (= 8 x 3) |

`embedding_scale` is forced to 1.0. The HF wrapper feeds `inputs_embeds`, so MiniCPM's
`scale_emb = 12` never applies, but llama.cpp scales injected embeddings by default.

The trajectory head ("funnel") is `LayerNorm -> 5 x (Linear + GELU) -> LayerNorm ->
Linear -> tanh -> * output_scale`, with `output_scale = [2, 2, 1]`.

Implementation: `src/models/robottrack.cpp`, subclassing `llama_model_minicpm`.

## Inference flow

Per frame:

1. Decode the frame (`mtmd_helper_bitmap_init_from_file` or `mtmd_helper_video_read_next`).
2. Push it into a 32-frame window; encode it to `coarse[4] + fine[64]`.
3. Build the history: the earlier frames' `coarse`, left-padded to 31 by repeating the
   oldest frame. A 1-frame window is its own history.
4. Assemble `[text | 31 x (marker + coarse[4]) | (marker + fine[64]) | control_query]`,
   giving `S = T + 155 + 65 + 1` rows of 1024 floats.
5. Feed it through `batch.embd` and call `llama_decode` once.
6. `llama_get_embeddings_seq()` returns the scaled `[8, 3]` trajectory.

Text embeddings, learned markers (`traj.*_embd`), and the control query are read raw
from the GGUF and assembled into `batch.embd` by the caller — the graph only sees the
final fused sequence. A marker row is `time[t] + stream[s] + camera[0]` (`stream=0`
for history, `stream=1` for the current frame).

Output routing: `build_pool` forces `LLAMA_POOLING_TYPE_RANK` to take the **last**
position (the trailing control query) instead of the first; `build_arch_head` then
feeds `t_embd_pooled` into the trajectory funnel.

Per-frame encoder output is cached. `--reencode` re-encodes the whole window every step,
the way the reference implementation does; results are bit-identical, and the flag exists
to check that.

## Conversion

The MiniCPM-RobotTrack checkpoint only contains the trajectory head and vision projector;
the two frozen vision towers — [DINOv3](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
and [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384) — need to be
downloaded separately and placed next to the RobotTrack directory:

```
model/
  MiniCPM-RobotTrack/
  dinov3-vits16-pretrain-lvd1689m/
  siglip-so400m-patch14-384/
```

Use `--dino-dir` / `--siglip-dir` to override the default sibling-directory lookup.

Both halves come from the same checkpoint directory, converted twice.

Backbone:

```bash
python convert_hf_to_gguf.py /path/to/MiniCPM-RobotTrack \
    --outfile robottrack-f16.gguf --outtype f16
```

`conversion/robottrack.py` maps `trajectory_head.layers.*` to
`traj.{norm_in,fc1..fc5,norm_out,out}`, `temporal_markers.*` to `traj.*_embd`, and skips
`vision_projector.*`. `traj.*` stays F32.

mmproj:

```bash
python convert_hf_to_gguf.py /path/to/MiniCPM-RobotTrack --mmproj --outtype f32 \
    --dino-dir /path/to/dinov3-vits16-pretrain-lvd1689m \
    --siglip-dir /path/to/siglip-so400m-patch14-384
```

## Build

```bash
cmake -S . -B build
cmake --build build --target llama-minicpm-robottrack -j8
```

The example links `mtmd`, which is defined under `tools/mtmd`, so it is only built when
`LLAMA_BUILD_TOOLS` is on.

## Usage

```
llama-minicpm-robottrack <mmproj.gguf> <model.gguf> "<instruction>" <input> [options]
```

`<input>` is dispatched by content:

| input | handling |
|---|---|
| directory | `*.jpg` then `*.png`, each sorted |
| image file | single frame |
| anything else | video, via ffmpeg/ffprobe in `PATH` |

| option | default | meaning |
|---|---|---|
| `--max-frames N` | 0 (all) | stop after N frames |
| `--threads N` | 8 | backbone threads |
| `--vision-threads N` | 4 | mmproj threads |
| `--fps F` | 0 (native) | video target fps |
| `--control-waypoint K` | 1 | waypoint used for the printed velocity |
| `--control-dt D` | 0.1 | control timestep for the printed velocity |
| `--reencode` | off | re-encode the whole window each step |

Example:

```bash
build/bin/llama-minicpm-robottrack \
    minicpm-robottrack-mmproj-f32.gguf robottrack-f16.gguf \
    "Follow the person." /path/to/frames
```

Output is one line per frame, plus the full waypoint table for single-image input:

```
[text] T=5 ids=1 22046 1358 2787 72
[input] frame directory /path/to/frames (64 frames)
[seq] S=226 (T=5 + 155 hist + 65 curr + 1 control)
  frame      0/64 N=1  frame_00001.jpg        vel=(+0.6657, -0.0893, -0.4749)
  frame      1/64 N=2  frame_00002.jpg        vel=(+0.4676, -0.1312, -0.4964)
```

`vel` is `trajectory[control_waypoint] / control_dt`.
