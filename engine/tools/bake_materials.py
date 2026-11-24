#!/usr/bin/env python3
"""Bake scanned diffuse maps into the raw RGBA8 files the engine loads.

Input: CC0 diffuse scans (for example Poly Haven *_diff_*.jpg). Output, per material:
  <name>_albedo_<size>.rgba  sRGB-encoded colour (the shader linearises it)
  <name>_normal_<size>.rgba  tangent-space normal, n * 0.5 + 0.5 in RGB; alpha = height (parallax)

No normal map was supplied with these scans, so the normal map is derived from a
luminance height proxy with wrap-around differences, which keeps the tile seamless.
Raw files are row 0 at the top, tightly packed, so the engine needs no image decoder.

Usage: python3 tools/bake_materials.py wood=wood_diff.jpg rock=rock_diff.jpg --out assets
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageFilter

# Per-material bump strength: how strongly luminance changes tilt the normal.
STRENGTH = {"wood": 3.0, "rock": 5.0}


def bake(name, src, out_dir, size):
    img = Image.open(src).convert("RGB").resize((size, size), Image.LANCZOS)
    albedo = np.asarray(img, dtype=np.uint8)
    alpha = np.full((size, size, 1), 255, np.uint8)
    albedo_rgba = np.concatenate([albedo, alpha], axis=-1)

    lum = np.asarray(img.convert("L").filter(ImageFilter.GaussianBlur(1.0)), np.float32) / 255.0
    dx = 0.5 * (np.roll(lum, -1, axis=1) - np.roll(lum, 1, axis=1))
    dy = 0.5 * (np.roll(lum, -1, axis=0) - np.roll(lum, 1, axis=0))
    s = STRENGTH.get(name, 4.0)
    n = np.stack([-s * dx, -s * dy, np.ones_like(lum)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    normal_rgb = np.clip((n * 0.5 + 0.5) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    # Height for parallax occlusion: the luminance proxy, centred on its mean and scaled by its spread
    # (about 0.5 = mean, 0 and 1 at four standard deviations). Stored in the normal map's alpha.
    height = np.clip(0.5 + (lum - lum.mean()) / (4.0 * lum.std() + 1e-6), 0.0, 1.0)
    height_u8 = np.clip(height * 255.0 + 0.5, 0, 255).astype(np.uint8)[..., None]
    normal_rgba = np.concatenate([normal_rgb, height_u8], axis=-1)

    a_path = os.path.join(out_dir, f"{name}_albedo_{size}.rgba")
    n_path = os.path.join(out_dir, f"{name}_normal_{size}.rgba")
    albedo_rgba.tofile(a_path)
    normal_rgba.tofile(n_path)
    mean = albedo.reshape(-1, 3).mean(axis=0)
    print(f"{name}: mean sRGB {mean.round(1)} -> {a_path}, {n_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", help="name=path pairs")
    p.add_argument("--out", default="assets")
    p.add_argument("--size", type=int, default=512)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for item in args.inputs:
        name, src = item.split("=", 1)
        bake(name, src, args.out, args.size)


if __name__ == "__main__":
    main()
