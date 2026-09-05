import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

HAS_TORCH=bool(importlib.util.find_spec('torch') and importlib.util.find_spec('torchvision'))


@unittest.skipUnless(HAS_TORCH,'Install the deep extra for neural-model tests')
class DeepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        cfg={'device_id':'d','protocol_id':'p','synthetic':True,'wavelengths_nm':list(range(500,590,10)),
             'wavelength_tolerance_nm':.1,'min_reference_dn':10,'min_valid_fraction':.95,
             'min_roi_pixels':100,'max_median_transmission':1.2,'rgb_size':64,
             'max_rgb_saturated_fraction':.05,'min_green_dn':10,'clahe_clip':2.,'clahe_grid':[8,8]}
        (cls.root/'config.json').write_text(json.dumps(cfg))
        rows=[];rng=np.random.default_rng(12)
        mask=np.zeros((32,32),dtype='uint8');mask[4:28,4:28]=255
        cv2.imwrite(str(cls.root/'mask.png'),mask)
        for split in ['train','val','calib','test']:
            for i in range(8):
                egg=f'{split}{i}';label=i%2
                raw=110+label*15+rng.normal(0,1,(32,32,9))
                np.savez(cls.root/f'{egg}.npz',raw=raw,dark=np.full(9,10),white=np.full(9,510),
                         wavelengths_nm=np.arange(500,590,10),mask=mask>0,saturation_dn=4095)
                rgb=np.zeros((32,32,3),dtype='uint8');rgb[mask>0]=[50,90+label*20,150]
                cv2.imwrite(str(cls.root/f'{egg}.png'),rgb)
                rows.append(dict(egg_id=egg,batch_id=f'{split}_{i//4}',sex='F' if label else 'M',split=split,
                                 device_id='d',protocol_id='p',hsi_path=f'{egg}.npz',rgb_path=f'{egg}.png',rgb_mask_path='mask.png'))
        cls.df=pd.DataFrame(rows);cls.df.to_csv(cls.root/'manifest.csv',index=False)
        cls.df[cls.df.split=='test'].to_csv(cls.root/'test.csv',index=False)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_all_branches_receive_gradients(self):
        import torch
        from eggsex.deep import EggNet
        for mode in ['hsi','rgb','fusion']:
            model=EggNet(mode)
            pred=model(torch.randn(2,3,9),torch.randn(2,3,64,64))
            self.assertEqual(tuple(pred.shape),(2,));pred.sum().backward()
            if mode!='rgb':self.assertGreater(model.spectral.layers[0].weight.grad.abs().sum(),0)
            if mode!='hsi':self.assertGreater(model.image.conv1.weight.grad.abs().sum(),0)

    def test_train_checkpoint_reload_and_test_isolation(self):
        from eggsex.deep import train_deep,predict_deep
        # Test assets intentionally invalid during training: training must not open them.
        df=self.df.copy();df.loc[df.split=='test','hsi_path']='not_present.npz'
        df.to_csv(self.root/'train_manifest.csv',index=False)
        for mode in ['hsi','fusion']:
            run=self.root/f'run_{mode}'
            result=train_deep(self.root/'train_manifest.csv',self.root/'config.json',run,
                              mode=mode,epochs=1,batch_size=4,size=64)
            self.assertFalse(result['test_evaluated'])
            a=predict_deep(run/'model.pt',self.root/'test.csv',run/'test',evaluate=True)
            b=predict_deep(run/'model.pt',self.root/'test.csv',run/'repeat',evaluate=True)
            self.assertEqual(a['n'],8);self.assertEqual(a['accuracy'],b['accuracy'])
            pa=pd.read_csv(run/'test/predictions.csv').p_female
            pb=pd.read_csv(run/'repeat/predictions.csv').p_female
            np.testing.assert_allclose(pa,pb,rtol=0,atol=0)
            with self.assertRaisesRegex(ValueError,'overlaps'):
                predict_deep(run/'model.pt',self.root/'manifest.csv',run/'bad',evaluate=True)
            invalid=self.df[self.df.split=='test'].copy();invalid.device_id='other'
            invalid.to_csv(self.root/'invalid.csv',index=False)
            empty=predict_deep(run/'model.pt',self.root/'invalid.csv',run/'invalid',evaluate=True)
            self.assertEqual(empty['n'],0)

    def test_image_variants_have_fixed_shape(self):
        from eggsex.deep import image_array
        cfg=json.loads((self.root/'config.json').read_text())
        for variant in ['rgb','green','clahe']:
            arr=image_array(self.df.iloc[0],self.root,cfg,variant,64)
            self.assertEqual(arr.shape,(3,64,64));self.assertTrue(np.isfinite(arr).all())

    def test_full_grouped_cv_keeps_final_test_unopened(self):
        from eggsex.benchmark import run as classical_cv
        from eggsex.deep_cv import run as neural_cv
        df=self.df.copy();df.loc[df.split=='test','hsi_path']='does_not_exist.npz'
        df.to_csv(self.root/'cv_manifest.csv',index=False)
        classical=classical_cv(self.root/'cv_manifest.csv',self.root/'config.json',self.root/'classical_cv',
                               modes=('hsi','rgb','fusion'),models=('logreg',),seeds=(42,),outer_n=3,inner_n=2,
                               permutations=1,bootstraps=50)
        self.assertEqual(classical['common_cohort_n'],24)
        self.assertEqual(len(classical['results']),3)
        self.assertEqual(classical['test_eggs_not_opened'],8)
        self.assertIn('permutation',classical['results'][0])
        deep=neural_cv(self.root/'cv_manifest.csv',self.root/'config.json',self.root/'neural_cv',
                       modes=('hsi',),seeds=(42,),outer_n=3,epochs=1,size=64,batch_size=4)
        self.assertEqual(deep['common_cohort_n'],24)
        self.assertEqual(deep['final_test_eggs_not_opened'],8)
        pred=pd.read_csv(self.root/'neural_cv/oof_predictions.csv')
        self.assertEqual(len(pred),24);self.assertFalse(pred.egg_id.duplicated().any())
