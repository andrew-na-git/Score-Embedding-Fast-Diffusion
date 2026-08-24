"""Spatio-temporal score network for 2D dynamic video.

The name says 3d because the convolutions are `Conv3d`; the data is 2D video,
(T, C, H, W). No volumetric support is claimed or implemented.

`network/network.py` is a 2D DDPM UNet. Rather than duplicate it, this module
reuses its building blocks and inserts temporal mixing, so the spatial pathway
stays identical to the published architecture and the temporal additions are the
only difference under ablation.

Design
------
(2+1)D factorisation: each spatial residual block is grouped with an optional
temporal convolution over the frame axis and optional spatial/temporal attention
into a single `SpatioTemporalBlock`. Grouping matters -- the UNet skip stack takes
one tensor per block, and a flat list of layers silently corrupts it.

Factorised rather than full 3D convolution because it keeps the spatial weights
shape-compatible with a pretrained 2D checkpoint: see `inflate_from_2d`, which
lets the video model start from an already-fitted image model.

Temporal layers are zero-initialised at their output projection, so at
initialisation the network is exactly the 2D model applied per frame. "No temporal
layers" is therefore a genuine ablation baseline, not a different model.

Conditioning
------------
The image pipeline conditions on an `nn.Embedding(num_images)` identity lookup,
which cannot generalise beyond the fitted instances. That is retained as
`clip_emb` for the per-instance fitting regime, but `frame_emb` is added so the
network knows *where in the clip* it is -- required by the autoregressive sampler
in `fast_diffusion/model/sample_video.py`.
"""

import functools

import torch
import torch.nn as nn

from .layers import (
    AttnBlock, Downsample, ResnetBlockDDPM, Upsample, ddpm_conv3x3, default_init,
    get_timestep_embedding_fourier, get_timestep_embedding_linear,
)


def zero_module(module):
    """Zero a module's parameters so it is initially a no-op."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class TemporalConv(nn.Module):
    """Depth-preserving 1D convolution over the frame axis.

    Input is (B*T, C, H, W) with `n_frames` supplied. Residual with a
    zero-initialised output projection.
    """

    def __init__(self, channels, kernel_size=3, dropout=0.0):
        super().__init__()
        padding = kernel_size // 2
        self.norm = nn.GroupNorm(num_channels=channels, num_groups=32, eps=1e-6)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = zero_module(
            nn.Conv1d(channels, channels, kernel_size, padding=padding)
        )

    def forward(self, x, n_frames):
        bt, c, h, w = x.shape
        b = bt // n_frames

        y = self.norm(x)
        # (B*T, C, H, W) -> (B*H*W, C, T)
        y = y.reshape(b, n_frames, c, h, w).permute(0, 3, 4, 2, 1).reshape(-1, c, n_frames)
        y = self.conv2(self.dropout(self.act(self.conv1(y))))
        y = y.reshape(b, h, w, c, n_frames).permute(0, 4, 3, 1, 2).reshape(bt, c, h, w)
        return x + y


class TemporalAttnBlock(nn.Module):
    """Self-attention across frames, applied independently per spatial site."""

    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.norm = nn.GroupNorm(num_channels=channels, num_groups=32, eps=1e-6)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.proj = zero_module(nn.Linear(channels, channels))

    def forward(self, x, n_frames):
        bt, c, h, w = x.shape
        b = bt // n_frames

        y = self.norm(x)
        # (B*T, C, H, W) -> (B*H*W, T, C): each spatial site is a sequence over T.
        y = y.reshape(b, n_frames, c, h, w).permute(0, 3, 4, 1, 2).reshape(-1, n_frames, c)
        y, _ = self.attn(y, y, y, need_weights=False)
        y = self.proj(y)
        y = y.reshape(b, h, w, n_frames, c).permute(0, 3, 4, 1, 2).reshape(bt, c, h, w)
        return x + y


class SpatioTemporalBlock(nn.Module):
    """One residual block plus its optional temporal and attention refinements.

    Emits a single tensor, which is what the UNet skip stack expects.
    """

    def __init__(self, res_block, temporal_conv=None, spatial_attn=None, temporal_attn=None):
        super().__init__()
        self.res = res_block
        self.tconv = temporal_conv
        self.attn = spatial_attn
        self.tattn = temporal_attn

    def forward(self, x, temb, n_frames):
        h = self.res(x, temb)
        if self.tconv is not None:
            h = self.tconv(h, n_frames)
        if self.attn is not None:
            h = self.attn(h)
        if self.tattn is not None:
            h = self.tattn(h, n_frames)
        return h


class VideoNet(nn.Module):
    """UNet score network over clips.

    Accepts `x` of shape (B, T, C, H, W), or (B*T, C, H, W) with `n_frames` given.
    Returns the score field in the same layout as the input.
    """

    def __init__(self, config):
        super().__init__()
        self.act = act = nn.SiLU()

        mcfg = config["model"]
        self.max_positions = mcfg["max_positions"]
        self.temb_func = (
            get_timestep_embedding_linear
            if mcfg["embedding_method"] == "linear"
            else get_timestep_embedding_fourier
        )
        self.nf = nf = mcfg["ch"]
        ch_mult = mcfg["ch_mult"]
        num_res_blocks = mcfg["num_res_blocks"]
        attn_res = mcfg["attention_resolutions"]
        tattn_res = mcfg.get("temporal_attention_resolutions", [])
        use_tconv = mcfg.get("temporal_conv", True)
        dropout = mcfg["dropout"]
        resamp_with_conv = mcfg["resample_with_conv"]
        num_resolutions = len(ch_mult)
        self.num_resolutions = num_resolutions

        image_resolution = config["data_loader"]["image_size"]
        self.clip_len = config["data_loader"].get("clip_len", 16)
        all_resolutions = [image_resolution // (2 ** i) for i in range(num_resolutions)]
        self.all_resolutions = all_resolutions

        ResnetBlock = functools.partial(
            ResnetBlockDDPM, act=act, temb_dim=4 * nf, dropout=dropout
        )

        def make_block(in_ch, out_ch, resolution):
            return SpatioTemporalBlock(
                ResnetBlock(in_ch=in_ch, out_ch=out_ch),
                TemporalConv(out_ch, dropout=dropout) if use_tconv else None,
                AttnBlock(channels=out_ch) if resolution in attn_res else None,
                TemporalAttnBlock(out_ch) if resolution in tattn_res else None,
            )

        temb_layers = [nn.Linear(nf, nf * 4), nn.Linear(nf * 4, nf * 4)]
        for layer in temb_layers:
            layer.weight.data = default_init()(layer.weight.data.shape)
            nn.init.zeros_(layer.bias)
        self.temb_dense = nn.ModuleList(temb_layers)

        self.centered = False
        channels = mcfg.get("in_channels", 3)
        self.in_conv = ddpm_conv3x3(channels, nf)

        # Encoder: `down_spec` records ('block', i) or ('down', i) in order, so
        # forward can mirror the original stack discipline exactly.
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        self.down_spec = []

        hs_c, in_ch = [nf], nf
        for i_level in range(num_resolutions):
            for _ in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                self.down_spec.append(("block", len(self.down_blocks)))
                self.down_blocks.append(make_block(in_ch, out_ch, all_resolutions[i_level]))
                in_ch = out_ch
                hs_c.append(in_ch)
            if i_level != num_resolutions - 1:
                self.down_spec.append(("down", len(self.down_samples)))
                self.down_samples.append(
                    Downsample(channels=in_ch, with_conv=resamp_with_conv)
                )
                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        self.mid_block1 = ResnetBlock(in_ch=in_ch)
        self.mid_attn = AttnBlock(channels=in_ch)
        self.mid_tconv = TemporalConv(in_ch, dropout=dropout) if use_tconv else None
        self.mid_tattn = TemporalAttnBlock(in_ch)
        self.mid_block2 = ResnetBlock(in_ch=in_ch)

        # Decoder.
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        self.up_spec = []

        for i_level in reversed(range(num_resolutions)):
            for _ in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                self.up_spec.append(("block", len(self.up_blocks)))
                self.up_blocks.append(
                    make_block(in_ch + hs_c.pop(), out_ch, all_resolutions[i_level])
                )
                in_ch = out_ch
            if i_level != 0:
                self.up_spec.append(("up", len(self.up_samples)))
                self.up_samples.append(Upsample(channels=in_ch, with_conv=resamp_with_conv))
        assert not hs_c

        self.out_norm = nn.GroupNorm(num_channels=in_ch, num_groups=32, eps=1e-6)
        self.out_conv = ddpm_conv3x3(in_ch, mcfg.get("out_channels", 3), init_scale=0.0)

        self.clip_emb = nn.Embedding(
            mcfg.get("num_clips", mcfg.get("num_images", 1)), 4 * nf
        )
        self.frame_emb = nn.Embedding(mcfg.get("max_frames", 512), 4 * nf)
        nn.init.zeros_(self.frame_emb.weight)

    def _embed(self, labels, clip_idx, frame_idx, n_frames, device):
        temb = self.temb_func(labels, self.nf, self.max_positions)
        temb = self.temb_dense[0](temb)
        temb = self.temb_dense[1](self.act(temb))

        if clip_idx is not None:
            temb = temb + self.clip_emb(clip_idx.to(device))
        if frame_idx is None:
            reps = max(temb.shape[0] // n_frames, 1)
            frame_idx = torch.arange(n_frames, device=device).repeat(reps)
        return temb + self.frame_emb(frame_idx.to(device))

    def forward(self, x, labels, clip_idx=None, frame_idx=None, n_frames=None):
        reshape_back = x.dim() == 5
        if reshape_back:
            b, t = x.shape[:2]
            n_frames = t
            x = x.reshape(b * t, *x.shape[2:])
            if labels.numel() == b:
                labels = labels.repeat_interleave(t)
            if clip_idx is not None and clip_idx.numel() == b:
                clip_idx = clip_idx.repeat_interleave(t)
            if frame_idx is not None and frame_idx.numel() == t:
                # Frame indices are per-frame, not per-clip, so they tile across the
                # batch rather than repeat-interleaving like the clip index. The
                # flattened order is (b0t0..b0tT, b1t0..), so `repeat` is correct and
                # `repeat_interleave` would assign every frame in a clip the same
                # index. Without this the caller's (T,) frame_idx stays length T
                # while temb is length b*t and the addition fails outright.
                frame_idx = frame_idx.repeat(b)
        if n_frames is None:
            n_frames = self.clip_len

        temb = self._embed(labels, clip_idx, frame_idx, n_frames, x.device)
        h = x if self.centered else 2 * x - 1.0

        hs = [self.in_conv(h)]
        for kind, i in self.down_spec:
            if kind == "block":
                hs.append(self.down_blocks[i](hs[-1], temb, n_frames))
            else:
                hs.append(self.down_samples[i](hs[-1]))

        h = hs[-1]
        h = self.mid_block1(h, temb)
        h = self.mid_attn(h)
        if self.mid_tconv is not None:
            h = self.mid_tconv(h, n_frames)
        h = self.mid_tattn(h, n_frames)
        h = self.mid_block2(h, temb)

        for kind, i in self.up_spec:
            if kind == "block":
                h = self.up_blocks[i](torch.cat([h, hs.pop()], dim=1), temb, n_frames)
            else:
                h = self.up_samples[i](h)
        assert not hs

        h = self.out_conv(self.act(self.out_norm(h)))
        if reshape_back:
            h = h.reshape(b, t, *h.shape[1:])
        return h


def inflate_from_2d(video_net, state_dict):
    """Initialise `video_net` spatial weights from a fitted 2D `network.Net`.

    The 2D model stores every layer in a single flat `all_modules` ModuleList, so
    names do not correspond. Matching is therefore done positionally by shape over
    the spatial parameters, in declaration order.

    Temporal layers keep their zero-initialised values, so immediately after
    inflation the video model reproduces the 2D model applied per frame.

    Returns
    -------
    report : dict with `matched`, `unmatched_source`, `unmatched_target`.
    """
    src = [(k, v) for k, v in state_dict.items() if v.dim() > 0]
    own = video_net.state_dict()

    temporal_markers = ("tconv", "tattn", "mid_tconv", "mid_tattn", "frame_emb")
    targets = [
        k for k in own
        if not any(m in k for m in temporal_markers)
    ]

    used_src, matched = set(), 0
    for tgt in targets:
        for j, (name, tensor) in enumerate(src):
            if j in used_src or tensor.shape != own[tgt].shape:
                continue
            own[tgt] = tensor.clone()
            used_src.add(j)
            matched += 1
            break

    video_net.load_state_dict(own, strict=False)

    if matched == 0:
        raise RuntimeError("no spatial weights matched; check the source checkpoint")

    return {
        "matched": matched,
        "unmatched_source": len(src) - len(used_src),
        "unmatched_target": len(targets) - matched,
    }
