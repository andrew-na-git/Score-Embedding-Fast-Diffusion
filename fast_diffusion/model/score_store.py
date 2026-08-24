"""Storage for spatio-temporal score fields.

The image pipeline writes the whole score field as a single `scores.npy` of shape
(N, C, H, W). That does not carry over: at (N, C, T, H, W) a 16-frame 256x256 clip
with N=20 and C=3 is

    20 * 3 * 16 * 256 * 256 * 4 bytes = 251 MB

per clip, and the training loop in `train.py` currently holds the entire field in
memory as a tensor. Three strategies are provided.

raw        : uncompressed memory-mapped .npy. Same footprint on disk, but the
             training loop only pages in the timesteps it touches.
temporal   : truncated basis along the frame axis. The score field is smooth in
             time by construction wherever the sequential scheme is working, so
             a handful of components typically captures it. Lossy; the retained
             energy is reported so the compression can be defended.
quantised  : per-(timestep, channel) affine quantisation to int8. Fixed 4x
             reduction, cheap, and the error is straightforward to bound.

All three expose the same read interface, so the training loop is agnostic.
"""

import json
import os

import numpy as np


class ScoreStore:
    """Base interface. Subclasses implement `save` and `get`."""

    def __init__(self, path):
        self.path = path
        self.meta_path = os.path.splitext(path)[0] + ".meta.json"

    # -- to be implemented -------------------------------------------------

    def save(self, scores):
        raise NotImplementedError

    def get(self, timestep, channel=None):
        raise NotImplementedError

    # -- shared ------------------------------------------------------------

    def _write_meta(self, **kwargs):
        with open(self.meta_path, "w") as f:
            json.dump(kwargs, f, indent=2)

    def _read_meta(self):
        with open(self.meta_path) as f:
            return json.load(f)

    @property
    def nbytes_on_disk(self):
        total = os.path.getsize(self.path) if os.path.exists(self.path) else 0
        if os.path.exists(self.meta_path):
            total += os.path.getsize(self.meta_path)
        return total


class RawScoreStore(ScoreStore):
    """Uncompressed .npy, read back memory-mapped."""

    def save(self, scores):
        scores = np.asarray(scores, dtype=np.float32)
        np.save(self.path, scores)
        self._write_meta(kind="raw", shape=list(scores.shape))
        return self

    def open(self):
        self._array = np.load(self.path, mmap_mode="r")
        return self

    def get(self, timestep, channel=None):
        arr = getattr(self, "_array", None)
        if arr is None:
            self.open()
            arr = self._array
        return np.asarray(arr[timestep] if channel is None else arr[timestep, channel])


class TemporalBasisScoreStore(ScoreStore):
    """Truncated SVD along the frame axis.

    The field is reshaped to (N*C*H*W, T) and decomposed once; `rank` components
    are retained. Reconstruction is exact when `rank == T`.

    `retained_energy` in the metadata is the fraction of squared singular value
    mass kept. Report it -- a lossy score field needs that number attached.
    """

    def __init__(self, path, rank=4):
        super().__init__(path)
        self.rank = rank

    def save(self, scores):
        scores = np.asarray(scores, dtype=np.float32)
        N, C, T, H, W = scores.shape
        rank = min(self.rank, T)

        flat = np.moveaxis(scores, 2, -1).reshape(-1, T)
        U, S, Vt = np.linalg.svd(flat, full_matrices=False)

        coeff = (U[:, :rank] * S[:rank]).astype(np.float32)
        basis = Vt[:rank].astype(np.float32)

        energy = float((S[:rank] ** 2).sum() / max((S ** 2).sum(), 1e-12))
        np.savez(self.path, coeff=coeff, basis=basis)
        self._write_meta(
            kind="temporal",
            shape=[N, C, T, H, W],
            rank=rank,
            retained_energy=energy,
        )
        self.retained_energy = energy
        return self

    def open(self):
        data = np.load(self.path if self.path.endswith(".npz") else self.path + ".npz")
        self._coeff = data["coeff"]
        self._basis = data["basis"]
        self._meta = self._read_meta()
        return self

    def get(self, timestep, channel=None):
        if not hasattr(self, "_coeff"):
            self.open()

        N, C, T, H, W = self._meta["shape"]
        recon = (self._coeff @ self._basis).reshape(N, C, H, W, T)
        recon = np.moveaxis(recon, -1, 2)
        return recon[timestep] if channel is None else recon[timestep, channel]

    def reconstruction_error(self, scores):
        """Relative L2 error of the truncated reconstruction. Report this."""
        if not hasattr(self, "_coeff"):
            self.open()
        N, C, T, H, W = self._meta["shape"]
        recon = np.moveaxis((self._coeff @ self._basis).reshape(N, C, H, W, T), -1, 2)
        scores = np.asarray(scores, dtype=np.float32)
        return float(np.linalg.norm(recon - scores) / np.linalg.norm(scores))


class QuantisedScoreStore(ScoreStore):
    """Per-(timestep, channel) affine int8 quantisation. Fixed 4x reduction."""

    def save(self, scores):
        scores = np.asarray(scores, dtype=np.float32)
        N, C = scores.shape[:2]

        lo = scores.reshape(N, C, -1).min(axis=2)
        hi = scores.reshape(N, C, -1).max(axis=2)
        scale = np.maximum(hi - lo, 1e-12) / 255.0

        norm = (scores - lo[:, :, None, None, None]) / scale[:, :, None, None, None]
        codes = np.clip(np.round(norm), 0, 255).astype(np.uint8)

        np.savez(self.path, codes=codes, lo=lo, scale=scale)
        self._write_meta(kind="quantised", shape=list(scores.shape))
        return self

    def open(self):
        data = np.load(self.path if self.path.endswith(".npz") else self.path + ".npz")
        self._codes = data["codes"]
        self._lo = data["lo"]
        self._scale = data["scale"]
        return self

    def get(self, timestep, channel=None):
        if not hasattr(self, "_codes"):
            self.open()

        if channel is None:
            codes = self._codes[timestep].astype(np.float32)
            return codes * self._scale[timestep][:, None, None, None] + self._lo[timestep][:, None, None, None]

        codes = self._codes[timestep, channel].astype(np.float32)
        return codes * self._scale[timestep, channel] + self._lo[timestep, channel]

    def max_abs_error(self, scores):
        """Worst-case absolute reconstruction error, i.e. half a quantisation step."""
        if not hasattr(self, "_codes"):
            self.open()
        return float(self._scale.max() / 2)


_STORES = {
    "raw": RawScoreStore,
    "temporal": TemporalBasisScoreStore,
    "quantised": QuantisedScoreStore,
}


def get_score_store(path, kind="raw", **kwargs):
    """Build a score store. `kind` is 'raw', 'temporal' or 'quantised'."""
    if kind not in _STORES:
        raise ValueError(f"unknown store {kind!r}; expected one of {sorted(_STORES)}")
    return _STORES[kind](path, **kwargs)


def estimate_footprint(N, C, T, H, W, dtype_bytes=4):
    """Uncompressed score-field size in bytes. Call before allocating."""
    return N * C * T * H * W * dtype_bytes
