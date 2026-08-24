"""Video clip dataset for the spatio-temporal FP pipeline.

Mirrors the conventions of `data/Dataset.py` -- per-clip min/max normalisation to
[0, 1], bicubic resize, seeded selection -- but yields (T, C, H, W) clips instead
of single images, and exposes the metadata the FP solver needs (`n_data`,
`channels`, `image_res`, `clip_len`).

Backends
--------
folder : a directory per clip containing numbered frame images.
video  : video files decoded with torchvision.io.read_video (needs a working
         video backend, e.g. PyAV).
synthetic : procedurally generated motion. Deliberately included: it gives an
         exactly-known flow field and a controllable scene cut, so the ESS
         keyframe trigger and the warp-source ablation can be validated without
         depending on any dataset download.
"""

import math
import os
from abc import ABC, abstractmethod

import numpy as np
import torch
import torchvision

torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class VideoDataset(ABC):
    """Base class yielding (T, C, H, W) clips normalised to [0, 1]."""

    def __init__(self, image_res=64, clip_len=16, n_clips=1, seed=9, stride=1):
        self.image_res = image_res
        self.clip_len = clip_len
        self.n_data = n_clips
        self.stride = stride
        self.channels = 3

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(
                (image_res, image_res),
                interpolation=v2.InterpolationMode.BICUBIC,
                antialias=True,
            ),
        ])

        clips = self.load_clips()
        if len(clips) < n_clips:
            raise ValueError(f"requested {n_clips} clips but only {len(clips)} available")

        self.data = torch.stack([self._normalise(c) for c in clips[:n_clips]])

    @staticmethod
    def _normalise(clip):
        """Min/max normalise a clip as a whole.

        Per-clip rather than per-frame on purpose: normalising each frame
        independently injects brightness flicker that the temporal consistency
        metrics would then attribute to the model.
        """
        lo, hi = clip.min(), clip.max()
        if hi - lo < 1e-8:
            return torch.zeros_like(clip)
        return (clip - lo) / (hi - lo)

    @abstractmethod
    def load_clips(self):
        """Return a list of (T, C, H, W) float tensors, pre-normalisation."""

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        return self.data[idx]

    def frames(self, idx):
        """Iterate frames of clip `idx` as (C, H, W) tensors."""
        for t in range(self.data.shape[1]):
            yield self.data[idx, t]


class FolderVideoDataset(VideoDataset):
    """One subdirectory per clip, each holding numbered frame images."""

    def __init__(self, root, **kwargs):
        self.root = root
        super().__init__(**kwargs)

    def load_clips(self):
        from PIL import Image

        subdirs = sorted(
            d for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d))
        )

        clips = []
        for d in subdirs:
            path = os.path.join(self.root, d)
            files = sorted(
                f for f in os.listdir(path)
                if os.path.splitext(f)[1].lower() in _IMAGE_EXTS
            )[:: self.stride][: self.clip_len]

            if len(files) < self.clip_len:
                continue

            frames = [
                self.transform(Image.open(os.path.join(path, f)).convert("RGB"))
                for f in files
            ]
            clips.append(torch.stack(frames))
        return clips


class FileVideoDataset(VideoDataset):
    """Video files decoded with torchvision.io.read_video."""

    def __init__(self, paths, start_frame=0, **kwargs):
        self.paths = [paths] if isinstance(paths, str) else list(paths)
        self.start_frame = start_frame
        super().__init__(**kwargs)

    def load_clips(self):
        from torchvision.io import read_video

        clips = []
        for p in self.paths:
            frames, _, _ = read_video(p, output_format="TCHW")
            frames = frames[self.start_frame :: self.stride][: self.clip_len]
            if frames.shape[0] < self.clip_len:
                continue
            clips.append(torch.stack([self.transform(f) for f in frames]))
        return clips


class SyntheticVideoDataset(VideoDataset):
    """Procedural clips with known motion and an optional scene cut.

    Two moving Gaussian blobs on a textured background translate at a constant
    velocity, so the ground-truth flow is exact. When `scene_cut_at` is set, the
    pattern is resampled at that frame, producing a discontinuity that the keyframe
    trigger should detect. Use this as the controlled validation case.

    `contiguous` controls whether successive clips are the same scene continuing, or
    independent scenes. This distinction decides whether cross-clip score warm starts
    are meaningful at all, so it is not a cosmetic option:

    contiguous=False (default)
        Each clip resamples its pattern and colours, so clip k has nothing to do with
        clip k-1. The keyframe trigger correctly fires at every clip boundary and warm
        starts are correctly declined. This is the negative control.
    contiguous=True
        One long sequence chopped into consecutive windows: clip k starts where clip
        k-1 ended, with the same pattern and continuing motion. Warm starts apply, and
        this is the setting in which their benefit can be measured.
    """

    def __init__(self, velocity=(1.5, 0.75), scene_cut_at=None, contiguous=False,
                 **kwargs):
        self.velocity = velocity
        self.scene_cut_at = scene_cut_at
        self.contiguous = contiguous
        super().__init__(**kwargs)

    def _pattern(self, res, rng):
        yy, xx = np.mgrid[0:res, 0:res].astype(np.float32)
        base = 0.25 * (
            np.sin(2 * math.pi * xx / max(res / 6.0, 1))
            * np.cos(2 * math.pi * yy / max(res / 5.0, 1))
        )
        centres = rng.uniform(0.2, 0.8, size=(2, 2)) * res
        return base, centres

    def load_clips(self):
        res, T = self.image_res, self.clip_len
        vx, vy = self.velocity
        yy, xx = np.mgrid[0:res, 0:res].astype(np.float32)

        clips = []
        # In the contiguous case one pattern and one motion phase are shared by every
        # clip, and the frame counter keeps running across clip boundaries.
        shared = None
        if self.contiguous:
            rng = np.random.default_rng(1000)
            base, centres = self._pattern(res, rng)
            colours = rng.uniform(0.4, 1.0, size=(2, 3)).astype(np.float32)
            shared = (rng, base, centres, colours)

        for c in range(self.n_data):
            if self.contiguous:
                rng, base, centres, colours = shared
                t_offset = c * T
            else:
                rng = np.random.default_rng(1000 + c)
                base, centres = self._pattern(res, rng)
                colours = rng.uniform(0.4, 1.0, size=(2, 3)).astype(np.float32)
                t_offset = 0

            frames = []
            for t in range(T):
                # The cut index is measured in absolute frames, so with contiguous
                # clips a cut can fall at or inside any clip rather than only in the
                # first one.
                if self.scene_cut_at is not None and t + t_offset == self.scene_cut_at:
                    base, centres = self._pattern(res, rng)
                    colours = rng.uniform(0.4, 1.0, size=(2, 3)).astype(np.float32)
                    if self.contiguous:
                        shared = (rng, base, centres, colours)

                frame = np.repeat(base[None], 3, axis=0).copy()
                for b in range(centres.shape[0]):
                    cy = (centres[b, 0] + vy * (t + t_offset)) % res
                    cx = (centres[b, 1] + vx * (t + t_offset)) % res
                    blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (res / 10.0) ** 2)))
                    frame += colours[b][:, None, None] * blob[None]

                frames.append(torch.from_numpy(frame))
            clips.append(torch.stack(frames))
        return clips

    def true_flow(self):
        """Exact backward flow between consecutive frames, (2, H, W).

        Constant translation, so this is uniform. Use it to bound how much of the
        flow-source ablation gap is estimator error versus scheme behaviour.
        """
        vx, vy = self.velocity
        flow = torch.zeros(2, self.image_res, self.image_res)
        flow[0] = -vx
        flow[1] = -vy
        return flow


def get_video_dataset(config):
    """Build a video dataset from a run config.

    Reads `data_loader.{dataset, image_size, clip_len, num_clips, seed, stride,
    root, paths, scene_cut_at, velocity, contiguous}`.
    """
    dl = config["data_loader"]
    kind = dl.get("dataset", "synthetic")

    common = dict(
        image_res=dl.get("image_size", 64),
        clip_len=dl.get("clip_len", 16),
        n_clips=dl.get("num_clips", dl.get("num_images", 1)),
        seed=dl.get("seed", 9),
        stride=dl.get("stride", 1),
    )

    if kind == "synthetic":
        return SyntheticVideoDataset(
            velocity=tuple(dl.get("velocity", (1.5, 0.75))),
            scene_cut_at=dl.get("scene_cut_at"),
            contiguous=bool(dl.get("contiguous", False)),
            **common,
        )
    if kind == "folder":
        return FolderVideoDataset(root=dl["root"], **common)
    if kind == "video":
        return FileVideoDataset(
            paths=dl["paths"], start_frame=dl.get("start_frame", 0), **common
        )

    raise NotImplementedError(
        f"video dataset {kind!r} is not supported; expected 'synthetic', 'folder' or 'video'"
    )
