import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import folder_paths
import re
from einops import rearrange

# ==========================================
# 模型结构 (WAN 2.1 - 16ch, embed_dim=32)
# ==========================================
def normalization(channels):
    return nn.GroupNorm(32, channels)

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = normalization(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        h = self.norm(x)
        q = rearrange(self.q(h), "b c h w -> b 1 (h w) c")
        k = rearrange(self.k(h), "b c h w -> b 1 (h w) c")
        v = rearrange(self.v(h), "b c h w -> b 1 (h w) c")
        h = nn.functional.scaled_dot_product_attention(q, k, v)
        h = rearrange(h, "b 1 (h w) c -> b c h w", h=x.shape[-2], w=x.shape[-1])
        return x + self.proj_out(h)

class ResBlockEmb(nn.Module):
    def __init__(self, channels, emb_channels, dropout=0, out_channels=None):
        super().__init__()
        oc = out_channels or channels
        self.in_layers = nn.Sequential(
            normalization(channels), nn.SiLU(), nn.Conv2d(channels, oc, 3, 1, 1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, 2 * oc))
        self.out_norm = normalization(oc)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Dropout(dropout), zero_module(nn.Conv2d(oc, oc, 3, 1, 1)))
        self.skip = nn.Conv2d(channels, oc, 1) if oc != channels else nn.Identity()

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        sc, sh = torch.chunk(emb_out, 2, 1)
        h = self.out_norm(h) * (1 + sc) + sh
        return self.skip(x) + self.out_layers(h)

class LatentResizer(nn.Module):
    def __init__(self, in_channels=16, in_blocks=8, out_blocks=8,
                 channels=192, dropout=0.1, attn=False, embed_dim=32):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, channels, 3, 1, 1)
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

        self.in_blocks = nn.ModuleList()
        for i in range(in_blocks):
            if attn and (i == 1 or i == in_blocks - 1):
                self.in_blocks.append(AttnBlock(channels))
            self.in_blocks.append(ResBlockEmb(channels, embed_dim, dropout))

        self.out_blocks = nn.ModuleList()
        for i in range(out_blocks):
            if attn and (i == 1 or i == out_blocks - 1):
                self.out_blocks.append(AttnBlock(channels))
            self.out_blocks.append(ResBlockEmb(channels, embed_dim, dropout))

        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv2d(channels, in_channels, 3, 1, 1)

    def forward(self, x, scale=None, target_hw=None):
        if target_hw is not None:
            size = target_hw
        elif scale is not None:
            size = tuple(int(round(s * scale)) for s in x.shape[-2:])
        else:
            return x
        if size == x.shape[-2:]:
            return x

        emb_val = (scale - 1) if (scale is not None and target_hw is None) else 0.0
        emb = self.embed(torch.tensor([emb_val], dtype=x.dtype, device=x.device).unsqueeze(0))

        x = self.conv_in(x)
        for b in self.in_blocks:
            x = b(x, emb) if isinstance(b, ResBlockEmb) else b(x)
        x = F.interpolate(x, size=size, mode="bilinear")
        for b in self.out_blocks:
            x = b(x, emb) if isinstance(b, ResBlockEmb) else b(x)
        return self.conv_out(F.silu(self.norm_out(x)))

class VideoLatentResizer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.resizer = LatentResizer(**kwargs)

    def forward(self, x, scale=None, target_hw=None):
        B, C, T, H, W = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resizer(x, scale=scale, target_hw=target_hw)
        return rearrange(x, "(b t) c h w -> b c t h w", b=B, t=T)


# ==========================================
# 模型加载
# ==========================================
MODEL_CACHE = {}

def get_models_dir():
    try:
        return folder_paths.get_folder_paths("upscalers")[0]
    except:
        d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        os.makedirs(d, exist_ok=True)
        return d

def scan_models():
    files = []
    for ext in ("*.pth", "*.safetensors"):
        files.extend(glob.glob(os.path.join(get_models_dir(), ext)))
    names = sorted(os.path.basename(f) for f in files)
    return names if names else [f"(请将模型放入: {get_models_dir()})"]

def _load_raw_sd(path):
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        sd = load_file(path, device='cpu')
    else:
        sd = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']
    # FP8 自动转 FP16
    sd = {k: v.to(torch.float16) if v.dtype == torch.float8_e4m3fn else v
          for k, v in sd.items()}
    return sd

def _detect_arch(sd):
    cfg = {"in_channels": 16, "in_blocks": 8, "out_blocks": 8,
           "channels": 192, "dropout": 0.1, "attn": False, "embed_dim": 32}

    conv_key = 'resizer.conv_in.weight'
    if conv_key in sd:
        cfg["in_channels"] = sd[conv_key].shape[1]
        cfg["channels"] = sd[conv_key].shape[0]

    embed_key = 'resizer.embed.0.weight'
    if embed_key in sd:
        cfg["embed_dim"] = sd[embed_key].shape[0]

    in_res, out_res = set(), set()
    for k in sd:
        m = re.match(r'resizer\.in_blocks\.(\d+)\.in_layers\.', k)
        if m: in_res.add(int(m.group(1)))
        m = re.match(r'resizer\.out_blocks\.(\d+)\.in_layers\.', k)
        if m: out_res.add(int(m.group(1)))
    cfg["in_blocks"] = len(in_res)
    cfg["out_blocks"] = len(out_res)

    cfg["attn"] = any('.q.' in k for k in sd)
    return cfg

def _split_merged(sd):
    if not any(k.startswith("upscaler.") for k in sd):
        return sd, None
    up, down = {}, {}
    for k, v in sd.items():
        if k.startswith("upscaler."):
            up[k[len("upscaler."):]] = v
        elif k.startswith("downscaler."):
            down[k[len("downscaler."):]] = v
    return up, down

def load_model(name, mode, device):
    cache_key = f"{name}::{mode}"
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    path = os.path.join(get_models_dir(), name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型文件不存在: {path}")

    raw_sd = _load_raw_sd(path)
    up_sd, down_sd = _split_merged(raw_sd)
    model_sd = down_sd if (mode == "downscale" and down_sd) else up_sd
    cfg = _detect_arch(model_sd)

    model = VideoLatentResizer(**cfg)
    model.load_state_dict(model_sd, strict=False)
    model = model.to(device).eval()
    MODEL_CACHE[cache_key] = model
    return model


# ==========================================
# ComfyUI 节点
# ==========================================
class Wan21LatentResizerNode:
    """WAN 2.1 Latent 缩放节点。scale>1 放大，scale<1 缩小，scale=1 不处理。支持 FP8/FP16/合并模型。"""

    INPUT_TYPES = lambda: {
        "required": {
            "latent": ("LATENT",),
            "model_name": (scan_models(),),
            "scale": ("FLOAT", {"default": 2.5, "min": 0.1, "max": 10.0, "step": 0.1}),
            "device": (["cuda", "cpu"], {"default": "cuda"}),
            "use_fp16": ("BOOLEAN", {"default": False}),
        }
    }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "video/Wan21"

    def run(self, latent, model_name, scale, device, use_fp16):
        if model_name.startswith('('):
            raise ValueError("请将模型文件放入 upscalers 目录")

        if abs(scale - 1.0) < 1e-6:
            return (latent,)

        mode = "upscale" if scale > 1.0 else "downscale"
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        model = load_model(model_name, mode, dev)

        s = latent["samples"].clone()
        orig_dtype, orig_shape = s.dtype, s.shape
        if len(s.shape) == 4:
            s = s.unsqueeze(2)

        compute_dtype = torch.float16 if use_fp16 else torch.float32
        s = s.to(dev, compute_dtype)
        model = model.to(compute_dtype)

        with torch.no_grad():
            if mode == "upscale":
                out = model(s, scale=scale)
            else:
                B, C, T, H, W = s.shape
                th = max(1, int(round(H * scale)))
                tw = max(1, int(round(W * scale)))
                out = model(s, target_hw=(th, tw))

        if len(orig_shape) == 4:
            out = out.squeeze(2)
        out = out.cpu().to(orig_dtype)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return ({"samples": out},)


# ==========================================
# 注册
# ==========================================
NODE_CLASS_MAPPINGS = {
    "Wan21LatentResizerNode": Wan21LatentResizerNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan21LatentResizerNode": "WAN 2.1 Latent Resizer",
}
