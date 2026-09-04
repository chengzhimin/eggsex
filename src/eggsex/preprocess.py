"""Deterministic preprocessing for paired transmission images and spectral cubes."""
from pathlib import Path
import cv2
import numpy as np


def cube_features(path, cfg):
    with np.load(path, allow_pickle=False) as z:
        raw = z['raw'].astype(np.float32)
        dark = z['dark'].astype(np.float32)
        white = z['white'].astype(np.float32)
        wave = z['wavelengths_nm'].astype(float)
        mask = z['mask'].astype(bool)
        sat = float(z['saturation_dn'])
    if raw.ndim != 3 or mask.shape != raw.shape[:2] or mask.sum() < cfg['min_roi_pixels']:
        raise ValueError('cube shape/ROI invalid')
    if wave.shape != (raw.shape[-1],) or not np.isfinite(wave).all() or np.any(np.diff(wave) <= 0):
        raise ValueError('wavelengths must be finite, increasing, in nm')
    grid = np.asarray(cfg['wavelengths_nm'], float)
    if len(grid) < 3 or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0):
        raise ValueError('invalid configured wavelength grid')
    idx = np.abs(wave[:, None] - grid[None, :]).argmin(axis=0)
    if np.any(np.abs(wave[idx] - grid) > cfg['wavelength_tolerance_nm']) or len(set(idx)) != len(idx):
        raise ValueError('instrument wavelengths differ from model contract')
    dark = np.broadcast_to(dark, raw.shape)
    white = np.broadcast_to(white, raw.shape)
    r, d, w = raw[..., idx], dark[..., idx], white[..., idx]
    denominator = w - d
    if not np.isfinite(sat) or sat <= 0:
        raise ValueError('invalid saturation DN')
    valid = (np.isfinite(r) & np.isfinite(d) & np.isfinite(w) &
             (denominator > cfg['min_reference_dn']) & (r < sat) & (w < sat) & (r > d))
    valid_fraction = valid[mask].mean(axis=0)
    if valid_fraction.min() < cfg['min_valid_fraction']:
        raise ValueError('QC: excessive invalid reference pixels')
    transmission = np.full(r.shape, np.nan, dtype=np.float32)
    np.divide(r-d, denominator, out=transmission, where=valid)
    values = transmission[mask]
    if np.nanmedian(values) > cfg['max_median_transmission']:
        raise ValueError('QC: check transmission reference geometry')
    q25, median, q75 = np.nanpercentile(values, [25, 50, 75], axis=0)
    spectral_z = (median - median.mean()) / max(float(median.std()), 1e-8)
    features = np.r_[median, q75-q25, spectral_z]
    names = [f'cube_{kind}_{nm:g}nm' for kind in ['median', 'iqr', 'spectral_z'] for nm in grid]
    return features, names, {'min_valid_fraction': float(valid_fraction.min())}


def image_features(path, mask_path, cfg, preview_dir=None):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
        raise ValueError('image branch requires an 8-bit, three-channel image')
    mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_image is None or mask_image.shape != img.shape[:2]:
        raise ValueError('provide an aligned ROI mask')
    mask = mask_image > 0
    if mask.sum() < cfg['min_roi_pixels']:
        raise ValueError('ROI too small')
    g = img[..., 1]
    sat_fraction = float((img[mask].max(axis=1) >= 255).mean())
    if sat_fraction > cfg['max_rgb_saturated_fraction'] or np.median(g[mask]) < cfg['min_green_dn']:
        raise ValueError('QC: saturation or insufficient signal')
    yy, xx = np.where(mask)
    crop = np.s_[yy.min():yy.max()+1, xx.min():xx.max()+1]
    g, mask, img = g[crop], mask[crop], img[crop]
    n = cfg['rgb_size']
    scale = n / max(g.shape)
    size = (max(1, round(g.shape[1]*scale)), max(1, round(g.shape[0]*scale)))
    g = cv2.resize(g, size, interpolation=cv2.INTER_AREA)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask.astype('uint8'), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    g[~mask] = int(np.median(g[mask]))
    enhanced = cv2.createCLAHE(clipLimit=cfg['clahe_clip'], tileGridSize=tuple(cfg['clahe_grid'])).apply(g)
    interior = cv2.erode(mask.astype('uint8'), np.ones((5, 5), 'uint8')).astype(bool)
    if interior.sum() < 16:
        raise ValueError('ROI has no usable interior')
    gx = cv2.Sobel(enhanced.astype('float32')/255, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(enhanced.astype('float32')/255, cv2.CV_32F, 0, 1)
    magnitude = np.hypot(gx, gy)
    features, names = [], []
    for name, arr in [('blue', img[..., 0]/255), ('green', g/255), ('red', img[..., 2]/255), ('clahe', enhanced/255), ('gradient', magnitude)]:
        vals = arr[interior]
        features.extend([vals.mean(), vals.std(), *np.percentile(vals, [10, 25, 50, 75, 90])])
        names.extend([f'image_{name}_{s}' for s in ['mean','std','p10','p25','p50','p75','p90']])
    if preview_dir:
        out = Path(preview_dir); out.mkdir(parents=True, exist_ok=True)
        g[~mask] = 0; enhanced[~mask] = 0
        cv2.imwrite(str(out/'green.png'), g)
        cv2.imwrite(str(out/'clahe.png'), enhanced)
    return np.asarray(features), names, {'saturated_fraction': sat_fraction}


def extract(row, base, cfg, mode):
    if row['device_id'] != cfg['device_id'] or row['protocol_id'] != cfg['protocol_id']:
        raise ValueError('device/protocol outside model contract')
    parts, names, qc = [], [], {}
    if mode in ('hsi', 'fusion'):
        f, n, q = cube_features(base / row['hsi_path'], cfg)
        parts.append(f); names.extend(n); qc.update(q)
    if mode in ('rgb', 'fusion'):
        f, n, q = image_features(base / row['rgb_path'], base / row['rgb_mask_path'], cfg)
        parts.append(f); names.extend(n); qc.update(q)
    features = np.concatenate(parts)
    if not np.isfinite(features).all():
        raise ValueError('nonfinite extracted features')
    return features, names, qc
