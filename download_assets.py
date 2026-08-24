"""Download the external assets needed for an apples-to-apples evaluation.

Two things are fetched, and both are required for comparability rather than
convenience:

I3D (Kinetics-400), for FVD
    FVD is backbone-sensitive: the same clips scored with a different feature
    extractor give a different number, so an FVD computed with a substitute backbone
    cannot be compared with any published value. The canonical extractor is the
    Kinetics-400 I3D used by Unterthiner et al. The TorchScript export mirrored by
    the StyleGAN-V authors is the de facto standard in PyTorch reimplementations and
    is what is fetched here.

DAVIS 2017 (480p trainval), for the inpainting task
    The standard benchmark for video object removal, and the reason it matters here
    is that it ships *real* per-object segmentation masks. Synthetic boxes make the
    task easier in a way that is hard to quantify; real object masks are irregular,
    track real motion, and are what the video-inpainting literature reports on. 90
    sequences is also enough to clear the FVD sample-count floor.

Usage
-----
    python download_assets.py                 # both
    python download_assets.py --only i3d
    python download_assets.py --only davis
    python download_assets.py --verify        # re-check what is already present

Everything lands in `assets/` which is gitignored; the recorded SHA-256 of each
download is written to `assets/MANIFEST.json` so a later run can prove it is using
the same bytes.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
import zipfile

ASSET_DIR = "assets"

ASSETS = {
    "i3d": {
        "url": "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1",
        "path": os.path.join(ASSET_DIR, "i3d_torchscript.pt"),
        "expected_bytes": 51235320,
        "note": "Kinetics-400 I3D, TorchScript. Canonical FVD feature extractor.",
    },
    "davis": {
        "url": "https://data.vision.ee.ethz.ch/csergi/share/davis/"
               "DAVIS-2017-trainval-480p.zip",
        "path": os.path.join(ASSET_DIR, "DAVIS-2017-trainval-480p.zip"),
        "expected_bytes": 832766765,
        "note": "DAVIS 2017 480p trainval: 90 sequences with real object masks.",
        "extract_to": os.path.join(ASSET_DIR, "DAVIS"),
    },
}


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download(url, dest, expected_bytes=None):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as fh:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        last = 0.0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            now = time.time()
            if now - last > 2.0:
                pct = f"{done/total*100:5.1f}%" if total else "  ?  "
                rate = done / max(now - t0, 1e-9)
                sys.stdout.write(f"\r    {pct}  {human(done)}  {human(rate)}/s   ")
                sys.stdout.flush()
                last = now
    sys.stdout.write("\r" + " " * 60 + "\r")

    got = os.path.getsize(tmp)
    if expected_bytes and got != expected_bytes:
        os.remove(tmp)
        raise RuntimeError(
            f"size mismatch for {url}: got {got} bytes, expected {expected_bytes}. "
            "The remote artifact changed; do not proceed with a different file "
            "silently, since FVD values depend on the exact weights."
        )
    os.replace(tmp, dest)
    return got, time.time() - t0


def sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_i3d(path):
    """Load the TorchScript module and confirm it yields the expected feature size.

    FVD is defined on the 400-dimensional Kinetics logits layer. Confirming the
    output width here means a later FVD run cannot silently be computing a Frechet
    distance over the wrong feature space.
    """
    import torch

    model = torch.jit.load(path).eval()
    # I3D expects (B, C, T, H, W) in [-1, 1] at 224x224 with at least ~9 frames.
    x = torch.zeros(1, 3, 16, 224, 224)
    with torch.no_grad():
        feats = model(x, rescale=False, resize=False, return_features=True)
    shape = tuple(feats.shape)
    if shape[-1] != 400:
        raise RuntimeError(
            f"I3D returned features of shape {shape}; FVD requires the 400-d "
            "Kinetics logits layer. Refusing to use these weights."
        )
    return shape


def verify_davis(zip_path, extract_to):
    """Extract and confirm the JPEG/Annotation structure DAVIS is expected to have."""
    if not os.path.isdir(extract_to):
        print(f"    extracting to {extract_to} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(ASSET_DIR)

    jpeg = os.path.join(extract_to, "JPEGImages", "480p")
    anno = os.path.join(extract_to, "Annotations", "480p")
    if not (os.path.isdir(jpeg) and os.path.isdir(anno)):
        raise RuntimeError(
            f"expected {jpeg} and {anno} after extraction; the archive layout is "
            "not what the DAVIS loader assumes"
        )
    seqs = sorted(os.listdir(jpeg))
    with_masks = [s for s in seqs if os.path.isdir(os.path.join(anno, s))]
    frames = sum(len(os.listdir(os.path.join(jpeg, s))) for s in seqs)
    return {"sequences": len(seqs), "with_masks": len(with_masks), "frames": frames}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=sorted(ASSETS), default=None)
    ap.add_argument("--verify", action="store_true",
                    help="verify what is already present without downloading")
    args = ap.parse_args()

    os.makedirs(ASSET_DIR, exist_ok=True)
    manifest_path = os.path.join(ASSET_DIR, "MANIFEST.json")
    manifest = {}
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)

    names = [args.only] if args.only else list(ASSETS)
    for name in names:
        spec = ASSETS[name]
        print(f"\n{name}: {spec['note']}")
        path = spec["path"]

        if os.path.isfile(path) and os.path.getsize(path) == spec["expected_bytes"]:
            print(f"    already present ({human(os.path.getsize(path))})")
        elif args.verify:
            print("    MISSING (run without --verify to download)")
            continue
        else:
            print(f"    downloading {human(spec['expected_bytes'])} ...")
            got, secs = download(spec["url"], path, spec["expected_bytes"])
            print(f"    downloaded {human(got)} in {secs:.0f}s")

        digest = manifest.get(name, {}).get("sha256")
        if not digest:
            print("    hashing ...")
            digest = sha256(path)
        print(f"    sha256 {digest[:16]}...")

        entry = {"url": spec["url"], "path": path,
                 "bytes": os.path.getsize(path), "sha256": digest}

        if name == "i3d":
            shape = verify_i3d(path)
            entry["feature_shape"] = list(shape)
            print(f"    verified: I3D features {shape} (400-d Kinetics logits)")
        elif name == "davis":
            stats = verify_davis(path, spec["extract_to"])
            entry.update(stats)
            entry["extracted_to"] = spec["extract_to"]
            print(f"    verified: {stats['sequences']} sequences, "
                  f"{stats['with_masks']} with masks, {stats['frames']} frames")

        manifest[name] = entry

    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nwrote {manifest_path}")


if __name__ == "__main__":
    main()
