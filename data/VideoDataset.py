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
    independent scenes. This decides whether the sequential density estimator's running
    grid stays valid across a clip boundary, so it is not a cosmetic option:

    contiguous=False (default)
        Each clip resamples its pattern and colours, so clip k has nothing to do with
        clip k-1. The KL trigger fires at every clip boundary and the density grid is
        correctly rebuilt from scratch. This is the negative control.
    contiguous=True
        One long sequence chopped into consecutive windows: clip k starts where clip
        k-1 ended, with the same pattern and continuing motion. The estimator's
        importance correction remains valid across boundaries, which is the regime its
        7.54x cost saving is measured in (`benchmark_keyframe_trigger.py`).
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


class DavisVideoDataset(VideoDataset):
    """DAVIS 2017 sequences with their real per-object segmentation masks.

    This is the dataset the inpainting comparison is meant to run on. Synthetic boxes
    and strokes (`inpaint.make_mask`) are useful for controlled ablations, but they
    make the task easier in a way that is hard to quantify: a box has straight edges,
    a fixed area and motion the model can extrapolate. DAVIS masks are irregular,
    track real objects, change area as objects rotate and occlude, and are what the
    video-inpainting literature reports on. 90 sequences also clears the FVD
    sample-count floor of 64 with one clip each.

    Layout expected under `root` (as unpacked by `download_assets.py`):

        JPEGImages/480p/<sequence>/00000.jpg ...
        Annotations/480p/<sequence>/00000.png ...
        ImageSets/2017/{train,val}.txt

    Masks are palette PNGs whose pixel value is the object id, 0 being background.
    DAVIS 2016-style binary annotations (0/255) are handled by the same code path
    since they are just the single-object case.

    Two things here are easy to get wrong and are done deliberately:

    * **Masks are resized with nearest-neighbour, never bicubic.** The frame
      transform uses bicubic, which is correct for images and wrong for masks: it
      produces fractional values along every boundary, so a mask that should be a
      hard 0/1 indicator becomes a soft one and the "known" region silently bleeds
      into the hole. Every masked metric would then be computed over the wrong
      support.
    * **Masks are not min/max normalised.** The base class normalises clips as a
      whole, which is right for pixels and meaningless for an indicator, so masks are
      kept out of `self.data` entirely.

    Parameters
    ----------
    root : DAVIS root directory.
    split : 'train', 'val' or 'all' -- which ImageSets list to read.
    year : '2017' or '2016', selecting the ImageSets subdirectory.
    objects : 'all' (union of every annotated object, the default), 'largest' (the
        single object with the greatest total area), or an int object id.
    mask_dilate : radius in pixels to grow the mask by. Video-inpainting papers
        commonly dilate, because an exactly-tight mask leaves a rim of the object's
        own colour just outside the hole which a method can copy from and appear to
        succeed. Default 0, so the choice is explicit rather than baked in.
    clip_start : first frame index within each sequence.
    min_coverage : sequences whose mask covers less than this fraction of the clip on
        average are skipped, which catches a sequence whose object leaves frame and,
        more usefully, an annotation directory that failed to load at all.
    """

    def __init__(self, root, split="all", year="2017", objects="all",
                 mask_dilate=0, clip_start=0, min_coverage=1e-4, **kwargs):
        self.root = root
        self.split = split
        self.year = str(year)
        self.objects = objects
        self.mask_dilate = int(mask_dilate)
        self.clip_start = int(clip_start)
        self.min_coverage = float(min_coverage)
        self._masks = []
        self._sequences = []
        self.skipped = []
        super().__init__(**kwargs)
        # `load_clips` fills `_masks` in the same order it returns clips; the base
        # class then truncates the clips to n_clips, so the masks must follow.
        self._masks = torch.stack(self._masks[: self.n_data])
        self._sequences = self._sequences[: self.n_data]

    def _sequence_names(self):
        if self.split == "all":
            names = sorted(os.listdir(os.path.join(self.root, "JPEGImages", "480p")))
            return [n for n in names
                    if os.path.isdir(os.path.join(self.root, "JPEGImages", "480p", n))]

        list_path = os.path.join(self.root, "ImageSets", self.year, f"{self.split}.txt")
        if not os.path.isfile(list_path):
            raise FileNotFoundError(
                f"DAVIS split list not found at {list_path!r}. Expected one of "
                f"'train', 'val' or 'all' for split, and a year with an ImageSets "
                f"directory."
            )
        with open(list_path, encoding="utf-8") as fh:
            return [ln.strip() for ln in fh if ln.strip()]

    def _load_mask(self, path, out_res):
        """One annotation PNG -> (1, H, W) float32 indicator, nearest-resized."""
        from PIL import Image

        ann = np.array(Image.open(path))
        if ann.ndim == 3:            # some tools save the palette expanded to RGB
            ann = ann[..., 0]

        ids = [int(v) for v in np.unique(ann) if v != 0]
        if not ids:
            sel = np.zeros_like(ann, dtype=bool)
        elif self.objects == "all":
            sel = ann != 0
        elif self.objects == "largest":
            areas = {i: int((ann == i).sum()) for i in ids}
            sel = ann == max(areas, key=areas.get)
        elif isinstance(self.objects, int):
            sel = ann == self.objects
        else:
            raise ValueError(
                f"objects must be 'all', 'largest' or an int id; got {self.objects!r}"
            )

        m = torch.from_numpy(sel.astype(np.float32))[None, None]  # (1,1,H,W)
        m = torch.nn.functional.interpolate(
            m, size=(out_res, out_res), mode="nearest")
        return m[0]

    def load_clips(self):
        from PIL import Image

        jpeg_root = os.path.join(self.root, "JPEGImages", "480p")
        ann_root = os.path.join(self.root, "Annotations", "480p")
        if not os.path.isdir(jpeg_root):
            raise FileNotFoundError(
                f"DAVIS frames not found at {jpeg_root!r}. Fetch and unpack the "
                "dataset with `python download_assets.py --only davis`."
            )

        need = self.clip_len * self.stride + self.clip_start
        clips = []

        for seq in self._sequence_names():
            frame_dir = os.path.join(jpeg_root, seq)
            ann_dir = os.path.join(ann_root, seq)
            if not os.path.isdir(frame_dir) or not os.path.isdir(ann_dir):
                self.skipped.append((seq, "missing frames or annotations"))
                continue

            files = sorted(f for f in os.listdir(frame_dir)
                           if os.path.splitext(f)[1].lower() in _IMAGE_EXTS)
            if len(files) < need:
                self.skipped.append(
                    (seq, f"only {len(files)} frames, need {need}"))
                continue

            picked = files[self.clip_start::self.stride][: self.clip_len]

            frames, masks = [], []
            for f in picked:
                frames.append(self.transform(
                    Image.open(os.path.join(frame_dir, f)).convert("RGB")))
                stem = os.path.splitext(f)[0]
                ann_path = os.path.join(ann_dir, stem + ".png")
                if not os.path.isfile(ann_path):
                    masks = []
                    break
                masks.append(self._load_mask(ann_path, self.image_res))

            if len(masks) != len(picked):
                self.skipped.append((seq, "annotation frame missing"))
                continue

            mask = torch.stack(masks)                  # (T, 1, H, W)
            coverage = float(mask.mean())
            if coverage < self.min_coverage:
                self.skipped.append((seq, f"mask coverage {coverage:.2e} too low"))
                continue

            if self.mask_dilate > 0:
                from fast_diffusion.model.inpaint import dilate_mask
                mask = torch.as_tensor(
                    dilate_mask(mask.numpy(), radius=self.mask_dilate),
                    dtype=torch.float32)

            clips.append(torch.stack(frames))
            self._masks.append(mask)
            self._sequences.append(seq)

        return clips

    def real_mask(self, idx):
        """Real object mask for clip `idx`, (T, 1, H, W) with 1 = hole.

        Same convention as `inpaint.make_mask`, so this drops straight into
        `pf_ode_inpaint` in place of a synthetic mask.
        """
        return self._masks[idx]

    def sequence_name(self, idx):
        return self._sequences[idx]

    def mask_report(self):
        """Per-clip mask coverage, plus which sequences were skipped and why.

        Coverage belongs next to any inpainting number: masked PSNR over a 2% hole
        and over a 30% hole are not remotely the same task.
        """
        cov = [float(m.mean()) for m in self._masks]
        return {
            "n_clips": len(cov),
            "sequences": list(self._sequences),
            "coverage_mean": float(np.mean(cov)) if cov else 0.0,
            "coverage_min": float(np.min(cov)) if cov else 0.0,
            "coverage_max": float(np.max(cov)) if cov else 0.0,
            "per_clip_coverage": cov,
            "n_skipped": len(self.skipped),
            "skipped": self.skipped,
        }


def get_video_dataset(config):
    """Build a video dataset from a run config.

    Reads `data_loader.{dataset, image_size, clip_len, num_clips, seed, stride,
    root, paths, scene_cut_at, velocity, contiguous, split, year, objects,
    mask_dilate, clip_start}`.
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
    if kind == "davis":
        return DavisVideoDataset(
            root=dl.get("root", os.path.join("assets", "DAVIS")),
            split=dl.get("split", "all"),
            year=dl.get("year", "2017"),
            objects=dl.get("objects", "all"),
            mask_dilate=dl.get("mask_dilate", 0),
            clip_start=dl.get("clip_start", 0),
            **common,
        )

    raise NotImplementedError(
        f"video dataset {kind!r} is not supported; expected 'synthetic', 'folder', "
        "'video' or 'davis'"
    )
