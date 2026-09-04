"""Explicit ENVI adapter; preserve measured wavelength/geometry metadata."""
import argparse
import cv2
import numpy as np
from spectral import envi


def load(path):
    img = envi.open(path)
    unit = str(img.metadata.get('wavelength units', '')).lower().strip()
    if unit not in ['nm','nanometers','nanometres','nanometer']:
        raise ValueError('Explicit nm wavelength units required; convert metadata deliberately')
    wave = np.asarray(img.metadata['wavelength'], float)
    return np.asarray(img.load(), dtype=np.float32), wave


def main():
    p = argparse.ArgumentParser()
    for name in ['raw','dark','white','mask','out']:
        p.add_argument('--'+name, required=True)
    p.add_argument('--saturation-dn', required=True, type=float)
    a = p.parse_args()
    raw, wave = load(a.raw)
    refs = []
    for path in [a.dark,a.white]:
        ref, w = load(path)
        if w.shape != wave.shape or not np.allclose(w, wave, rtol=0, atol=.01):
            raise ValueError('Reference wavelengths differ')
        if ref.shape != raw.shape:
            if ref.shape[1:] != raw.shape[1:]:
                raise ValueError('Reference cross-track geometry differs')
            ref = ref.mean(axis=0,keepdims=True)
        refs.append(ref)
    mask = cv2.imread(a.mask, cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != raw.shape[:2]:
        raise ValueError('Mask must be aligned to HSI cube')
    np.savez_compressed(a.out, raw=raw, dark=refs[0], white=refs[1], wavelengths_nm=wave,
                        mask=mask > 0, saturation_dn=np.array(a.saturation_dn))


if __name__ == '__main__':
    main()
