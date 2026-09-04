import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from eggsex.model import audit, select_threshold, metrics
from eggsex.preprocess import cube_features
from eggsex.envi_import import load
from spectral import envi


class PipelineTests(unittest.TestCase):
    def test_envi_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'cube.hdr'
            cube = np.arange(48, dtype='float32').reshape(4,4,3)
            envi.save_image(str(path), cube, metadata={'wavelength':[500,600,700], 'wavelength units':'nm'})
            actual, wave = load(str(path))
            np.testing.assert_allclose(actual, cube)
            np.testing.assert_allclose(wave, [500,600,700])

    def test_batch_leakage_is_rejected(self):
        rows = []
        for split in ['train','val','calib','test']:
            for sex in ['M','F']:
                rows.append(dict(egg_id=split+sex, batch_id='shared', split=split, sex=sex,
                                 device_id='d',protocol_id='p'))
        with self.assertRaisesRegex(ValueError, 'Batch leakage'):
            audit(pd.DataFrame(rows))

    def test_duplicate_egg_is_rejected(self):
        row = dict(egg_id='same',batch_id='b',device_id='d',protocol_id='p')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            audit(pd.DataFrame([row,row]), labeled=False)

    def test_bad_model_rejects_all(self):
        y = np.tile([0,1], 100); p = np.full(200, .5)
        self.assertGreater(select_threshold(y,p,.90,10), 1)
        self.assertEqual(metrics(y,p,1.1)['coverage'], 0)

    def test_cube_calibration_and_bad_reference(self):
        cfg = dict(min_roi_pixels=4,wavelengths_nm=[500,600,700],wavelength_tolerance_nm=.1,
                   min_reference_dn=20,min_valid_fraction=.95,max_median_transmission=1.2)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'a.npz'
            args = dict(raw=np.full((4,4,3),110), dark=np.full(3,10),white=np.full(3,210),
                        wavelengths_nm=np.array([500,600,700]),mask=np.ones((4,4),bool),saturation_dn=4095)
            np.savez(p,**args)
            f,_,_ = cube_features(p,cfg)
            np.testing.assert_allclose(f[:3], .5)
            args['white'] = args['dark']; np.savez(p,**args)
            with self.assertRaisesRegex(ValueError,'QC'):
                cube_features(p,cfg)

    def test_band_mismatch_is_rejected(self):
        cfg = dict(min_roi_pixels=4,wavelengths_nm=[510,600,700],wavelength_tolerance_nm=.1)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'a.npz'
            np.savez(p,raw=np.ones((4,4,3)),dark=np.zeros(3),white=np.ones(3),
                     wavelengths_nm=np.array([500,600,700]),mask=np.ones((4,4),bool),saturation_dn=4095)
            with self.assertRaisesRegex(ValueError,'wavelengths differ'):
                cube_features(p,cfg)


if __name__ == '__main__':
    unittest.main()
