import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import folder_paths
import re
from einops import rearrange

# ==========================================
# 模型结构
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
        self.in_layers = nn.Sequential(normalization(channels), nn.SiLU(), nn.Conv2d(channels, oc, 3, 1, 1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, 2 * oc))
        self.out_norm = normalization(oc)
        self.out_layers = nn.Sequential(nn.SiLU(), nn.Dropout(dropout), zero_module(nn.Conv2d(oc, oc, 3, 1, 1)))
        self.skip = nn.Conv2d(channels, oc, 1) if oc != channels else nn.Identity()

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, 1)
        h = self.out_norm(h) * (1 + scale) + shift
        return self.skip(x) + self.out_layers(h)

class LatentResizer(nn.Module):
    def __init__(self, in_channels=16, in_blocks=8, out_blocks=8, channels=192, dropout=0.1, attn=False):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, channels, 3, 1, 1)
        self.embed = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32))

        self.in_blocks = nn.ModuleList()
        for i in range(in_blocks):
            if attn and (i == 1 or i == in_blocks - 1):
                self.in_blocks.append(AttnBlock(channels))
            self.in_blocks.append(ResBlockEmb(channels, 32, dropout))

        self.out_blocks = nn.ModuleList()
        for i in range(out_blocks):
            if attn and (i == 1 or i == out_blocks - 1):
                self.out_blocks.append(AttnBlock(channels))
            self.out_blocks.append(ResBlockEmb(channels, 32, dropout))

        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv2d(channels, in_channels, 3, 1, 1)

    def forward(self, x, scale=None):
        size = tuple(int(round(s * scale)) for s in x.shape[-2:])
        if size == x.shape[-2:]:
            return x

        emb = self.embed(torch.tensor([scale - 1], dtype=x.dtype, device=x.device).unsqueeze(0))
        x = self.conv_in(x)
        for b in self.in_blocks:
            x = b(x, emb) if isinstance(b, ResBlockEmb) else b(x)
        
        x = F.interpolate(x, size=size, mode="bilinear")
        
        for b in self.out_blocks:
            x = b(x, emb) if isinstance(b, ResBlockEmb) else b(x)

        x = self.conv_out(F.silu(self.norm_out(x)))
        return x

class VideoLatentResizer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.resizer = LatentResizer(**kwargs)

    def forward(self, x, scale=None):
        B, C, T, H, W = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resizer(x, scale)
        return rearrange(x, "(b t) c h w -> b c t h w", b=B, t=T)

# ==========================================
# 模型加载
# ==========================================
MODEL_CACHE = {}

def get_models_dir():
    try:
        return folder_paths.get_folder_paths("upscalers")[0]
    except:
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        os.makedirs(d, exist_ok=True)
        return d

def scan_models():
    fs = glob.glob(os.path.join(get_models_dir(), "*.pth")) + glob.glob(os.path.join(get_models_dir(), "*.safetensors"))
    return [os.path.basename(f) for f in sorted(fs)] if fs else [f"(模型目录：{get_models_dir()})"]

def detect_model_architecture(sd):
    keys = set(sd.keys())
    new_arch = any('resizer.' in k for k in keys)
    cfg = {"in_channels":16,"in_blocks":8,"out_blocks":8,"channels":192,"dropout":0.1,"attn":False}

    if new_arch:
        if 'resizer.conv_in.weight' in sd:
            cfg["in_channels"] = sd['resizer.conv_in.weight'].shape[1]
            cfg["channels"] = sd['resizer.conv_in.weight'].shape[0]

        cfg["in_blocks"] = (len([k for k in keys if re.match(r'resizer\.in_blocks\.\d+\.', k)]) + 1) // 2
        cfg["out_blocks"] = (len([k for k in keys if re.match(r'resizer\.out_blocks\.\d+\.', k)]) + 1) // 2
        cfg["attn"] = any('AttnBlock' in k for k in keys)

    return VideoLatentResizer, cfg

def load_model(name, device):
    if name in MODEL_CACHE:
        m = MODEL_CACHE[name]
        return m.to(device) if next(m.parameters()).device != device else m

    path = os.path.join(get_models_dir(), name)
    if not os.path.exists(path): raise FileNotFoundError(path)

    try:
        if path.endswith('.safetensors'):
            from safetensors.torch import load_file
            sd = load_file(path, device='cpu')
        else:
            sd = torch.load(path, map_location='cpu', weights_only=True)
    except:
        sd = torch.load(path, map_location='cpu')

    cls, cfg = detect_model_architecture(sd)
    model = cls(**cfg)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    MODEL_CACHE[name] = model
    return model

# ==========================================
# ComfyUI 节点
# ==========================================
class Wan21LatentUpscalerNode:
    INPUT_TYPES = lambda: {
        "required": {
            "latent": ("LATENT",),
            "model_name": (scan_models(),),
            "scale": ("FLOAT", {"default":2.5,"min":0.1,"max":10,"step":0.1}),
            "device": (["cuda","cpu"],{"default":"cuda"}),
            "tile_size": ("INT",{"default":0,"min":0,"max":256,"step":32}),
        }
    }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "video/Wan21"

    def run(self, latent, model_name, scale, device, tile_size):
        if model_name.startswith('('): raise ValueError("请放入模型")
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        model = load_model(model_name, device)

        s = latent["samples"].clone()
        orig_dtype, orig_shape = s.dtype, s.shape
        if len(s.shape) == 4: s = s.unsqueeze(2)
        s = s.to(device, torch.float32)

        with torch.no_grad():
            out = self._tile(model, s, scale, tile_size) if tile_size else model(s, scale=scale)

        if len(orig_shape) == 4: out = out.squeeze(2)
        out = out.cpu().to(orig_dtype)
        if device.type == "cuda": torch.cuda.empty_cache()
        return ({"samples": out},)

    def _tile(self, model, x, scale, tile):
        B, C, T, H, W = x.shape
        oH, oW = int(H*scale), int(W*scale)
        out = torch.zeros(B, C, T, oH, oW, device=x.device, dtype=x.dtype)
        stride = tile // 2

        for h in range(0, H, stride):
            for w in range(0, W, stride):
                hh, ww = min(h+tile, H), min(w+tile, W)
                oh, ow = int(h*scale), int(w*scale)
                ohh, oww = int(hh*scale), int(ww*scale)
                t = model(x[:,:,:,h:hh,w:ww], scale)
                out[:,:,:,oh:ohh,ow:oww] = t
        return out

# ==========================================
# 注册（只保留主节点）
# ==========================================
NODE_CLASS_MAPPINGS = {
    "Wan21LatentUpscalerNode": Wan21LatentUpscalerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan21LatentUpscalerNode": "Wan2.1 Latent Resizer",
}