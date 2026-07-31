from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import LazyTorchTensor, ModelBase, MmprojModel, gguf, logger

from .minicpm import MiniCPMModel


@ModelBase.register("MiniCPMRobotTrackModel")
class MiniCPMRobotTrackModel(MiniCPMModel):
    """MiniCPM4 backbone + RobotTrack trajectory head in a single GGUF.

    The checkpoint bundles four things under one safetensors file:
      - backbone.*          the MiniCPM4-0.5B decoder (converted as usual)
      - temporal_markers.*  learned time/stream/camera marker tables
      - control_query       the learned query whose hidden state feeds the head
      - trajectory_head.*   the funnel MLP producing [num_waypoints, action_dim]
      - vision_projector.*  belongs to the mmproj, converted separately

    The head is an nn.Sequential; only indices 0/1/4/7/10/13/16/17 carry
    parameters (the rest are GELU/Dropout), which is why the funnel maps onto
    norm_in / fc1..fc5 / norm_out / out.
    """

    model_arch = gguf.MODEL_ARCH.ROBOTTRACK

    # trajectory_head.layers index -> gguf tensor
    _FUNNEL_MAP = {
        0:  gguf.MODEL_TENSOR.TRAJ_NORM_IN,
        1:  gguf.MODEL_TENSOR.TRAJ_FC1,
        4:  gguf.MODEL_TENSOR.TRAJ_FC2,
        7:  gguf.MODEL_TENSOR.TRAJ_FC3,
        10: gguf.MODEL_TENSOR.TRAJ_FC4,
        13: gguf.MODEL_TENSOR.TRAJ_FC5,
        16: gguf.MODEL_TENSOR.TRAJ_NORM_OUT,
        17: gguf.MODEL_TENSOR.TRAJ_OUT,
    }

    _MARKER_MAP = {
        "temporal_markers.time_embedding.weight":   gguf.MODEL_TENSOR.TRAJ_TIME_EMBD,
        "temporal_markers.stream_embedding.weight": gguf.MODEL_TENSOR.TRAJ_STREAM_EMBD,
        "temporal_markers.camera_embedding.weight": gguf.MODEL_TENSOR.TRAJ_CAMERA_EMBD,
    }

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        # The RobotTrack wrapper always feeds inputs_embeds, so MiniCPM's
        # scale_emb is never applied (modeling_minicpm.py only scales when
        # inputs_embeds is None). Bake that in rather than leaving callers to
        # override the KV at load time.
        self.gguf_writer.add_embedding_scale(1.0)
        logger.info("gguf: (robottrack) embedding_scale = 1.0 (wrapper feeds inputs_embeds)")

        cfg = self.hparams
        self.gguf_writer.add_uint32("robottrack.num_waypoints", cfg["num_waypoints"])
        self.gguf_writer.add_uint32("robottrack.action_dim", cfg["action_dim"])
        self.gguf_writer.add_uint32("robottrack.history_frames", cfg["history_frames"])
        self.gguf_writer.add_uint32("robottrack.coarse_tokens_per_frame", cfg["coarse_tokens_per_frame"])
        self.gguf_writer.add_uint32("robottrack.fine_tokens_current_frame", cfg["fine_tokens_current_frame"])
        self.gguf_writer.add_uint32("robottrack.max_time_steps", cfg["max_time_steps"])
        self.gguf_writer.add_bool("robottrack.use_tanh_actions", cfg["use_tanh_actions"])

        # the head output is num_waypoints * action_dim floats per sequence,
        # read back through the pooled-embedding path
        self.gguf_writer.add_classifier_output_labels(
            [f"wp{w}.{d}" for w in range(cfg["num_waypoints"]) for d in ("x", "y", "yaw")[: cfg["action_dim"]]]
        )

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # the mmproj half is converted by tools/mtmd, not here
        if name.startswith("vision_projector."):
            return

        if name in self._MARKER_MAP:
            yield (self.format_tensor_name(self._MARKER_MAP[name], suffix=".weight"), data_torch)
            return

        if name == "control_query":
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.TRAJ_CONTROL, suffix=".weight"), data_torch.reshape(-1))
            return

        if name == "output_scale":
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.TRAJ_OUT_SCALE, suffix=".weight"), data_torch.reshape(-1))
            return

        if name.startswith("trajectory_head.layers."):
            parts = name.split(".")
            idx, suffix = int(parts[2]), parts[3]
            tensor = self._FUNNEL_MAP.get(idx)
            if tensor is None:
                raise ValueError(f"unexpected parameter on a non-parametric funnel layer: {name}")
            yield (self.format_tensor_name(tensor, suffix=f".{suffix}"), data_torch)
            return

        # everything else is the backbone (or an already-formatted extra tensor
        # such as the rope factors emitted by generate_extra_tensors)
        yield from super().modify_tensors(data_torch, name.removeprefix("backbone."), bid)

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        # rope long/short factors live in backbone_config, already merged into hparams
        yield from super().generate_extra_tensors()

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        del name, bid, n_dims  # unused
        # Keep the whole trajectory head and the marker tables in F32. They are
        # tiny next to the backbone, the head output is only 24 floats wide so
        # quantization error shows up directly in the predicted trajectory, and
        # the marker tables are consumed by the caller when assembling the
        # sequence rather than by a matmul.
        if new_name.startswith("traj."):
            return gguf.GGMLQuantizationType.F32
        return False


@ModelBase.register("MiniCPMRobotTrackModel")
class MiniCPMRobotTrackVisionModel(MmprojModel):
    """mmproj half: DINOv3 + SigLIP dual tower and the RobotTrack VisionProjector.

    The checkpoint only carries `vision_projector.*`; both towers are stock
    pretrained encoders kept frozen, and config.json does not record which ones.
    They are looked up as siblings of the model directory, or via
    --dino-dir / --siglip-dir.

    Three tables the graph needs are computed here rather than at load time,
    since they depend only on image_size:
      - DINOv3 axial-RoPE cos/sin, [n_pos, head_dim], prefix rows left as identity
      - SigLIP s_grid^2 -> grid^2 adaptive average pool, as a dense matrix
      - per-tower mean / inv-std, so normalization runs in-graph on a /255 image
    """

    # RobotTrack resizes to 384 regardless of what the towers were trained at
    IMAGE_SIZE = 384

    DINO_DIR_NAME   = "dinov3-vits16-pretrain-lvd1689m"
    SIGLIP_DIR_NAME = "siglip-so400m-patch14-384"

    DINO_MEAN   = [0.485, 0.456, 0.406]
    DINO_STD    = [0.229, 0.224, 0.225]
    SIGLIP_MEAN = [0.5, 0.5, 0.5]
    SIGLIP_STD  = [0.5, 0.5, 0.5]

    # per-layer HF name -> gguf suffix, applied under v.dino.blk.{bid}. / v.siglip.blk.{bid}.
    _DINO_BLK = {
        "norm1":                 "ln1",
        "norm2":                 "ln2",
        "attention.q_proj":      "attn_q",
        "attention.k_proj":      "attn_k",
        "attention.v_proj":      "attn_v",
        "attention.o_proj":      "attn_out",
        "mlp.up_proj":           "ffn_up",
        "mlp.down_proj":         "ffn_down",
        "layer_scale1.lambda1":  "ls1.weight",
        "layer_scale2.lambda1":  "ls2.weight",
    }
    _SIGLIP_BLK = {
        "layer_norm1":         "ln1",
        "layer_norm2":         "ln2",
        "self_attn.q_proj":    "attn_q",
        "self_attn.k_proj":    "attn_k",
        "self_attn.v_proj":    "attn_v",
        "self_attn.out_proj":  "attn_out",
        "mlp.fc1":             "ffn_up",
        "mlp.fc2":             "ffn_down",
    }

    _DINO_TOP = {
        "embeddings.patch_embeddings.weight": "v.dino.patch_embd.weight",
        "embeddings.patch_embeddings.bias":   "v.dino.patch_embd.bias",
        "norm.weight":                        "v.dino.post_ln.weight",
        "norm.bias":                          "v.dino.post_ln.bias",
    }
    _SIGLIP_TOP = {
        "embeddings.patch_embedding.weight":    "v.siglip.patch_embd.weight",
        "embeddings.patch_embedding.bias":      "v.siglip.patch_embd.bias",
        "embeddings.position_embedding.weight": "v.siglip.position_embd.weight",
        "post_layernorm.weight":                "v.siglip.post_ln.weight",
        "post_layernorm.bias":                  "v.siglip.post_ln.bias",
    }

    # VisionProjector is an nn.Sequential; 2 is the GELU
    _PROJ = {
        "layers.0.weight": "mm.input_norm.weight",
        "layers.0.bias":   "mm.input_norm.bias",
        "layers.1.weight": "mm.fc1.weight",
        "layers.1.bias":   "mm.fc1.bias",
        "layers.3.weight": "mm.fc2.weight",
        "layers.3.bias":   "mm.fc2.bias",
    }

    @classmethod
    def _locate_tower(cls, dir_model: Path, name: str, override: str | None) -> Path:
        if override is not None:
            path = Path(override)
        else:
            path = dir_model.parent / name
        if not (path / "config.json").is_file():
            raise FileNotFoundError(
                f"{name} not found at {path}. The RobotTrack checkpoint ships only the "
                f"projector, so both vision towers have to be downloaded separately and "
                f"placed next to {dir_model}, or pointed at explicitly."
            )
        return path

    def __init__(self, dir_model: Path, *args: Any, **kwargs: Any):
        self.dino_dir   = self._locate_tower(dir_model, self.DINO_DIR_NAME,   kwargs.pop("dino_dir", None))
        self.siglip_dir = self._locate_tower(dir_model, self.SIGLIP_DIR_NAME, kwargs.pop("siglip_dir", None))

        with open(self.dino_dir / "config.json", "r", encoding="utf-8") as f:
            dino_cfg = json.load(f)
        with open(self.siglip_dir / "config.json", "r", encoding="utf-8") as f:
            siglip_cfg = json.load(f)
        self.hparams_siglip = siglip_cfg.get("vision_config", siglip_cfg)

        # MmprojModel needs a vision_config to initialise; the DINOv3 tower drives the
        # generic clip.vision.* geometry and the SigLIP tower gets its own key prefix
        if kwargs.get("hparams") is None:
            hparams = ModelBase.load_hparams(dir_model, False)
            hparams["vision_config"] = {**dino_cfg, "image_size": self.IMAGE_SIZE}
            kwargs["hparams"] = hparams

        super().__init__(dir_model, *args, **kwargs)

        # both towers normalize in-graph, so the preprocessor only does /255
        self.preprocessor_config = {"image_mean": [0.0, 0.0, 0.0], "image_std": [1.0, 1.0, 1.0]}

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        # everything else in this checkpoint is the backbone / trajectory head
        return (name, gen) if name.startswith("vision_projector.") else None

    def _index_tower(self, path: Path, prefix: str) -> dict[str, Callable[[], Tensor]]:
        out: dict[str, Callable[[], Tensor]] = {}
        with gguf.utility.SafetensorsLocal(path / "model.safetensors") as part:
            for name in part.keys():
                if prefix == "siglip." and not name.startswith("vision_model."):
                    continue  # the SigLIP checkpoint also carries the text tower
                data = part[name]
                if self.lazy:
                    gen = lambda data=data: LazyTorchTensor.from_local_tensor(data)  # noqa: E731
                else:
                    dtype = LazyTorchTensor._dtype_str_map[data.dtype]
                    gen = lambda data=data, dtype=dtype: torch.from_numpy(data.mmap_bytes()).view(dtype).reshape(data.shape)  # noqa: E731
                out[prefix + name.removeprefix("vision_model.")] = gen
        return out

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        tensors = super().index_tensors(remote_hf_model_id)
        tensors.update(self._index_tower(self.dino_dir,   "dino."))
        tensors.update(self._index_tower(self.siglip_dir, "siglip."))
        return tensors

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.MINICPM_TRACK)
        self.gguf_writer.add_vision_attention_layernorm_eps(self.find_vparam(["layer_norm_eps"]))

        s = self.hparams_siglip
        self.gguf_writer.add_uint32("clip.vision.siglip.embedding_length", s["hidden_size"])
        self.gguf_writer.add_uint32("clip.vision.siglip.feed_forward_length", s["intermediate_size"])
        self.gguf_writer.add_uint32("clip.vision.siglip.block_count", s["num_hidden_layers"])
        self.gguf_writer.add_uint32("clip.vision.siglip.patch_size", s["patch_size"])
        self.gguf_writer.add_uint32("clip.vision.siglip.attention.head_count", s["num_attention_heads"])
        self.gguf_writer.add_float32("clip.vision.siglip.attention.layer_norm_epsilon",
                                     s.get("layer_norm_eps", 1e-6))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("v.") or name.startswith("mm."):
            yield (name, data_torch)  # already-formatted table from generate_extra_tensors
            return

        if name.startswith("vision_projector."):
            yield (self._PROJ[name.removeprefix("vision_projector.")], data_torch)
            return

        if name.startswith("dino."):
            body = name.removeprefix("dino.")
            if body == "embeddings.cls_token":
                yield ("v.dino.class_embd", data_torch.reshape(-1))
            elif body == "embeddings.register_tokens":
                yield ("v.dino.register_embd", data_torch.reshape(-1, data_torch.shape[-1]))
            elif body == "embeddings.mask_token":
                return  # unused by the graph
            elif body in self._DINO_TOP:
                yield (self._DINO_TOP[body], data_torch)
            else:
                yield (self._map_block("v.dino", body, "layer.", self._DINO_BLK), data_torch)
            return

        if name.startswith("siglip."):
            body = name.removeprefix("siglip.")
            if body.startswith("head."):
                return  # attention-pooling head, unused: RobotTrack reads patch tokens
            if body in self._SIGLIP_TOP:
                yield (self._SIGLIP_TOP[body], data_torch)
            else:
                yield (self._map_block("v.siglip", body, "encoder.layers.", self._SIGLIP_BLK), data_torch)
            return

        raise ValueError(f"unexpected mmproj tensor {name!r}")

    @staticmethod
    def _map_block(prefix: str, body: str, layer_prefix: str, table: dict[str, str]) -> str:
        if not body.startswith(layer_prefix):
            raise ValueError(f"unexpected tensor {body!r}")
        idx, _, rest = body.removeprefix(layer_prefix).partition(".")
        for hf, gg in table.items():
            if rest == hf:                       # layer_scale, already carries .weight
                return f"{prefix}.blk.{idx}.{gg}"
            if rest.startswith(hf + "."):
                return f"{prefix}.blk.{idx}.{gg}{rest[len(hf):]}"
        raise ValueError(f"unexpected tensor {body!r}")

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        d_embd  = self.find_vparam(["hidden_size"])
        n_head  = self.find_vparam(["num_attention_heads"])
        d_head  = d_embd // n_head
        grid    = self.IMAGE_SIZE // self.find_vparam(["patch_size"])
        s_grid  = self.IMAGE_SIZE // self.hparams_siglip["patch_size"]
        prefix  = 1 + self.find_vparam(["num_register_tokens"])

        cos, sin = self._dino_rope(d_head, grid, prefix, self.find_vparam(["rope_theta"]))
        yield ("v.dino.rope_cos", torch.from_numpy(cos))
        yield ("v.dino.rope_sin", torch.from_numpy(sin))
        yield ("v.siglip.pool", torch.from_numpy(self._siglip_pool(s_grid, grid)))

        for tag, mean, std in (("dino",   self.DINO_MEAN,   self.DINO_STD),
                               ("siglip", self.SIGLIP_MEAN, self.SIGLIP_STD)):
            yield (f"v.{tag}.image_mean",    torch.tensor(mean, dtype=torch.float32))
            yield (f"v.{tag}.image_inv_std", torch.tensor(1.0 / np.array(std), dtype=torch.float32))

    @staticmethod
    def _dino_rope(head_dim: int, grid: int, num_prefix: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
        """Axial 2D RoPE matching transformers DINOv3ViTRopePositionEmbedding."""
        inv_freq = 1.0 / theta ** np.arange(0, 1, 4.0 / head_dim, dtype=np.float64)
        coords_1d = (np.arange(grid, dtype=np.float64) + 0.5) / grid
        hh, ww = np.meshgrid(coords_1d, coords_1d, indexing="ij")
        coords = 2.0 * np.stack([hh, ww], axis=-1).reshape(-1, 2) - 1.0
        angles = 2.0 * math.pi * coords[:, :, None] * inv_freq[None, None, :]
        angles = np.tile(angles.reshape(coords.shape[0], -1), 2)

        n_pos = num_prefix + coords.shape[0]
        cos = np.ones((n_pos, head_dim), dtype=np.float64)    # prefix rows: cos=1
        sin = np.zeros((n_pos, head_dim), dtype=np.float64)   # prefix rows: sin=0
        cos[num_prefix:] = np.cos(angles)
        sin[num_prefix:] = np.sin(angles)
        return (np.ascontiguousarray(cos, dtype=np.float32),
                np.ascontiguousarray(sin, dtype=np.float32))

    @staticmethod
    def _siglip_pool(n_in: int, n_out: int) -> np.ndarray:
        """Adaptive average pool n_in^2 -> n_out^2, as a dense [n_out^2, n_in^2] matrix."""
        a = np.zeros((n_out, n_in), dtype=np.float64)
        for o in range(n_out):
            start = (o * n_in) // n_out
            end = -(-((o + 1) * n_in) // n_out)  # ceil
            a[o, start:end] = 1.0 / (end - start)
        return np.ascontiguousarray(np.kron(a, a), dtype=np.float32)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        del name, bid, n_dims  # unused
        # the baked tables are geometry, not weights; the rest follows --outtype
        if new_name in ("v.dino.rope_cos", "v.dino.rope_sin", "v.siglip.pool"):
            return gguf.GGMLQuantizationType.F32
        return False
