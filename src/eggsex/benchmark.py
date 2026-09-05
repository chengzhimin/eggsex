"""Nested grouped CV on development eggs only, with a paired common QC cohort."""
import json
from pathlib import Path
from zipfile import BadZipFile
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .model import audit
from .preprocess import extract
from .evaluation import scores, cluster_interval, permute_within_batch
from sklearn.svm import SVC


class PLSDA(ClassifierMixin, BaseEstimator):
    """Binary 0/1 PLS regression with a classification interface."""
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, x, y):
        self.classes_ = np.array([0,1])
        self.n_features_in_ = x.shape[1]
        n = min(self.n_components, x.shape[1], x.shape[0]-1)
        self.pls_ = PLSRegression(n_components=n, scale=False).fit(x,y)
        return self

    def decision_function(self, x):
        return self.pls_.predict(x).ravel()-.5

    def predict(self, x):
        return (self.decision_function(x) >= 0).astype(int)


def candidate(name, seed):
    if name == 'logreg':
        model = LogisticRegression(max_iter=2000, class_weight='balanced',random_state=seed)
        grid = {'model__C':[.1,1.,10.]}
    elif name == 'svm':
        model = SVC(class_weight='balanced')
        grid = {'model__C':[.1,1.,10.]}
    elif name == 'plsda':
        model = PLSDA()
        grid = {'model__n_components':[2,5]}
    elif name == 'extra_trees':
        model = ExtraTreesClassifier(n_estimators=150, class_weight='balanced',random_state=seed,n_jobs=1)
        grid = {'model__max_depth':[3,None]}
    else:
        raise ValueError(f'Unknown estimator: {name}')
    return Pipeline([('scale',StandardScaler()),('model',model)]), grid


def group_folds(x, y, groups, n, seed):
    if len(np.unique(groups)) < n or n < 2:
        raise ValueError(f'Need at least {n} independent groups')
    folds = list(GroupKFold(n_splits=n,shuffle=True,random_state=seed).split(x,y,groups))
    for tr, va in folds:
        if len(np.unique(y[tr])) != 2 or len(np.unique(y[va])) != 2:
            raise ValueError('A grouped fold lacks one sex; revise batch design/fold count')
    return folds


def nested_predict(x, y, groups, name, seed, outer_n, inner_n):
    pred = np.full(len(y),np.nan); fold_id = np.full(len(y),-1)
    selected = []
    outer = group_folds(x,y,groups,outer_n,seed)
    for i, (tr, va) in enumerate(outer):
        inner = group_folds(x[tr],y[tr],groups[tr],inner_n,seed+1)
        estimator, grid = candidate(name,seed)
        search = GridSearchCV(estimator, grid, cv=inner,scoring='roc_auc',error_score='raise',n_jobs=1)
        search.fit(x[tr],y[tr])
        # Explicit grouped calibration folds: no hidden random image-level CV.
        model = CalibratedClassifierCV(clone(search.best_estimator_),cv=inner,method='sigmoid')
        model.fit(x[tr],y[tr])
        pred[va] = model.predict_proba(x[va])[:,1]; fold_id[va] = i
        selected.append({'fold':i,'parameters':search.best_params_,'inner_auc':float(search.best_score_)})
    if not np.isfinite(pred).all():
        raise RuntimeError('Incomplete out-of-fold predictions')
    return pred, fold_id, selected


def run(manifest, config, out, modes=('hsi','rgb','fusion'),
        models=('logreg','plsda','svm','extra_trees'), seeds=(42,123,2026),
        outer_n=5, inner_n=3, permutations=0, bootstraps=1000):
    manifest, out = Path(manifest).resolve(), Path(out)
    if permutations < 0 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Use nonnegative permutations and distinct seeds')
    out.mkdir(parents=True,exist_ok=True)
    cfg = json.loads(Path(config).read_text())
    original = pd.read_csv(manifest,dtype=str,keep_default_na=False); audit(original)
    # No test image/cube is opened during exploratory model selection.
    df = original[original.split != 'test'].reset_index(drop=True)
    features = {m:[] for m in modes}; keep, rejects = [], []
    for i,row in df.iterrows():
        try:
            items = {m:extract(row,manifest.parent,cfg,m)[0] for m in modes}
            for m in modes:
                features[m].append(items[m])
            keep.append(i)
        except (OSError,ValueError,KeyError,BadZipFile) as exc:
            rejects.append({'egg_id':row.egg_id,'reason':str(exc)})
    df = df.loc[keep].reset_index(drop=True)
    if df.empty:
        raise ValueError('No common valid cohort')
    y = (df.sex == 'F').to_numpy(dtype=int); groups = df.batch_id.to_numpy()
    pd.DataFrame(rejects,columns=['egg_id','reason']).to_csv(out/'qc_rejected.csv',index=False)
    df.to_csv(out/'cohort.csv',index=False)
    all_predictions, summaries = [], []
    for mode in modes:
        x = np.stack(features[mode])
        for name in models:
            for seed in seeds:
                p, fold, params = nested_predict(x,y,groups,name,seed,outer_n,inner_n)
                report = scores(y,p)
                report.update(mode=mode,model=name,seed=int(seed),parameters=params,
                              accuracy_batch_ci=cluster_interval(y,p,groups,bootstraps,seed))
                if permutations:
                    rng = np.random.default_rng(seed)
                    null = []
                    for _ in range(permutations):
                        yp = permute_within_batch(y,groups,rng)
                        pp,_,_ = nested_predict(x,yp,groups,name,seed,outer_n,inner_n)
                        null.append(scores(yp,pp)['auc'])
                    report['permutation'] = {'n':permutations,'statistic':'OOF AUC',
                        'p_value':float((1+np.sum(np.asarray(null) >= report['auc']))/(permutations+1)),
                        'null_auc':null,'note':'within-batch, full nested procedure rerun; unadjusted across comparisons'}
                summaries.append(report)
                rec = df[['egg_id','batch_id','sex']].copy()
                rec['mode']=mode;rec['model']=name;rec['seed']=seed;rec['fold']=fold;rec['p_female']=p
                all_predictions.append(rec)
    pd.concat(all_predictions).to_csv(out/'oof_predictions.csv',index=False)
    payload = {'evaluation':'nested batch CV; development only','test_eggs_not_opened':int((original.split == 'test').sum()),
               'synthetic':bool(cfg.get('synthetic',False)),'common_cohort_n':len(df),
               'outer_folds':outer_n,'inner_folds':inner_n,'results':summaries,
               'note':'Repeated seeds reuse eggs. Do not treat folds/seeds as independent subjects or their scores as a final blinded test.'}
    (out/'benchmark.json').write_text(json.dumps(payload,indent=2,allow_nan=False))
    pd.DataFrame([{k:r[k] for k in ['mode','model','seed','n','accuracy','balanced_accuracy','auc','macro_f1','mcc','brier']}
                  for r in summaries]).to_csv(out/'summary.csv',index=False)
    return payload
