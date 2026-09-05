import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from eggsex.evaluation import scores,cluster_interval,permute_within_batch,compare_files
from eggsex.benchmark import nested_predict,group_folds


class ResearchTests(unittest.TestCase):
    def test_single_sex_metrics_are_defined(self):
        r=scores([1,1],[.8,.7])
        self.assertIsNone(r['auc']);self.assertIsNone(r['by_sex']['M']['recall'])
        self.assertEqual(r['accuracy'],1)

    def test_invalid_probabilities_rejected(self):
        with self.assertRaisesRegex(ValueError,'probabilities'):
            scores([0,1],[.2,1.2])

    def test_cluster_pair_is_zero_for_identical_predictions(self):
        y=np.tile([0,1],8);p=np.linspace(.1,.9,len(y));g=np.repeat(np.arange(4),4)
        r=cluster_interval(y,p,g,repetitions=100,other=p)
        self.assertEqual(r['ci95'],[0.,0.])

    def test_permutation_preserves_each_batch_sex_counts(self):
        y=np.tile([0,0,1,1],6);groups=np.repeat(np.arange(6),4)
        p=permute_within_batch(y,groups,np.random.default_rng(1))
        for g in np.unique(groups):
            self.assertEqual(sum(y[groups==g]),sum(p[groups==g]))

    def test_outer_folds_have_no_group_overlap(self):
        y=np.tile([0,1],12);x=np.arange(48).reshape(24,2);g=np.repeat(np.arange(6),4)
        for tr,va in group_folds(x,y,g,3,42):
            self.assertFalse(set(g[tr])&set(g[va]))

    def test_nested_model_families_return_finite_oof(self):
        rng=np.random.default_rng(3);y=np.tile([0,1],18);g=np.repeat(np.arange(6),6)
        x=rng.normal(size=(len(y),8));x[:,0]+=y*2
        for model in ['logreg','plsda','svm','extra_trees']:
            p,fold,_=nested_predict(x,y,g,model,42,3,2)
            self.assertTrue(np.isfinite(p).all());self.assertEqual(set(fold),{0,1,2})

    def test_comparison_rejects_mislabeled_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a.csv';b=Path(tmp)/'b.csv'
            df=pd.DataFrame({'egg_id':['a','b'],'batch_id':['1','2'],'sex':['M','F'],'p_female':[.2,.8]})
            df.to_csv(a,index=False);df['sex']=['F','M'];df.to_csv(b,index=False)
            with self.assertRaisesRegex(ValueError,'disagree'):
                compare_files(a,b,Path(tmp)/'out.json')
