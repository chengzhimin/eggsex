"""Synthetic engineering test. Injected labels have no biological interpretation."""
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from .model import train, predict


def run(out):
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    wave = np.arange(500, 801, 10).astype(float)
    cfg = {'device_id':'SYNTHETIC', 'protocol_id':'DEMO_ONLY', 'synthetic':True,
           'age_min_h':84, 'age_max_h':96, 'wavelengths_nm':wave.tolist(),
           'wavelength_tolerance_nm':0.1, 'min_reference_dn':20, 'min_valid_fraction':.95,
           'min_roi_pixels':100, 'max_median_transmission':1.2, 'rgb_size':256,
           'max_rgb_saturated_fraction':.01, 'min_green_dn':20,
           'clahe_clip':2., 'clahe_grid':[8,8],
           'validation_target_lower_bound':.90, 'min_accepted_per_sex':10}
    (out/'config.json').write_text(json.dumps(cfg, indent=2))
    yy, xx = np.mgrid[:40,:40]; mask = (xx-20)**2+(yy-20)**2 < 16**2
    cv2.imwrite(str(out/'mask.png'), mask.astype('uint8')*255)
    rows = []
    for j, split in enumerate(['train','val','calib','test']):
        for i in range(80):
            label = i%2; egg = f'demo_{j}_{i:03}'
            curve = .2+.04*label*np.exp(-((wave-600)/45)**2)+rng.normal(0,.002,len(wave))
            raw = 100+curve[None,None,:]*2000+rng.normal(0,2,(40,40,len(wave)))
            np.savez_compressed(out/f'{egg}.npz', raw=raw.astype('float32'),
                                dark=np.full(len(wave),100), white=np.full(len(wave),2100),
                                wavelengths_nm=wave, mask=mask, saturation_dn=np.array(4095))
            rgb = np.zeros((40,40,3), dtype='uint8')
            rgb[mask] = [60,100+20*label,180]
            cv2.line(rgb,(20,8),(20,31),(20,30,90),1)
            cv2.imwrite(str(out/f'{egg}.png'),rgb)
            rows.append({'egg_id':egg,'batch_id':f'batch_{j}', 'sex':'F' if label else 'M',
                         'split':split,'device_id':'SYNTHETIC','protocol_id':'DEMO_ONLY','age_h':90,
                         'hsi_path':f'{egg}.npz','rgb_path':f'{egg}.png','rgb_mask_path':'mask.png'})
    df = pd.DataFrame(rows); df.to_csv(out/'manifest.csv',index=False)
    df[df.split == 'test'].drop(columns=['sex','split']).to_csv(out/'inference.csv', index=False)
    for mode in ['hsi','rgb','fusion']:
        train(out/'manifest.csv',out/'config.json',out/mode, mode)
        predict(out/mode/'model.joblib',out/'inference.csv',out/mode/'inference.csv')
    print('SYNTHETIC ONLY: extraction, training, calibration, test and inference completed.')
