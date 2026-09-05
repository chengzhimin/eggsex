"""Egg-level scores and batch-cluster resampling for fixed predictions."""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, roc_auc_score,
                             average_precision_score, brier_score_loss, log_loss,
                             confusion_matrix, matthews_corrcoef, f1_score)


def scores(y, probability):
    y = np.asarray(y, dtype=int); p = np.asarray(probability, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y) or not set(y).issubset({0,1}):
        raise ValueError('Expected nonempty binary egg labels and matching probabilities')
    if not np.isfinite(p).all() or np.any((p < 0)|(p > 1)):
        raise ValueError('Invalid probabilities')
    pred = (p >= .5).astype(int)
    cm = confusion_matrix(y, pred, labels=[0,1])
    per_sex = {}
    for i, name in enumerate(['M','F']):
        per_sex[name] = {'n':int(cm[i].sum()),
                         'recall':float(cm[i,i]/cm[i].sum()) if cm[i].sum() else None,
                         'precision':float(cm[i,i]/cm[:,i].sum()) if cm[:,i].sum() else None}
    bins = np.minimum((p*10).astype(int), 9)
    ece = sum(np.mean(bins == b)*abs(float(p[bins == b].mean()-y[bins == b].mean()))
              for b in range(10) if np.any(bins == b))
    both = len(np.unique(y)) == 2
    return {'n':len(y), 'accuracy':float(accuracy_score(y,pred)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)) if both else None,
            'auc':float(roc_auc_score(y,p)) if both else None,
            'average_precision_female':float(average_precision_score(y,p)) if both else None,
            'macro_f1':float(f1_score(y,pred,labels=[0,1],average='macro',zero_division=0)),
            'mcc':float(matthews_corrcoef(y,pred)) if both else None, 'brier':float(brier_score_loss(y,p)),
            'log_loss':float(log_loss(y,p,labels=[0,1])), 'ece_10_bins':float(ece),
            'confusion_matrix_M_F':cm.tolist(), 'by_sex':per_sex}


def cluster_interval(y, p, groups, repetitions=1000, seed=42, other=None):
    """Percentile CI of accuracy, or paired accuracy difference p - other.

    Resamples whole batches, not folds or repeated images. It conditions on
    existing predictions; it does not refit models or correct model selection.
    """
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    if y.shape != p.shape or groups.shape != y.shape or repetitions < 1:
        raise ValueError('Invalid resampling input')
    if other is not None and np.asarray(other).shape != y.shape:
        raise ValueError('Paired predictions must be aligned by egg_id')
    keys = np.unique(groups)
    if len(keys) < 2:
        return {'ci95':None,'batches':len(keys),'reason':'at least two batches required'}
    correct = ((p >= .5) == y).astype(float)
    if other is not None:
        correct -= ((np.asarray(other) >= .5) == y).astype(float)
    # Resampling aggregate counts equals concatenating the full sampled batches.
    counts = np.array([(groups == k).sum() for k in keys])
    sums = np.array([correct[groups == k].sum() for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(keys), size=(repetitions,len(keys)))
    draws = sums[idx].sum(axis=1)/counts[idx].sum(axis=1)
    return {'estimate':float(correct.mean()),'ci95':np.quantile(draws,[.025,.975]).tolist(),
            'batches':len(keys),'repetitions':repetitions,
            'note':'fixed-prediction batch bootstrap; not a model-selection-adjusted CI'}


def permute_within_batch(y, groups, rng):
    y, groups = np.asarray(y), np.asarray(groups)
    shuffled = y.copy()
    for group in np.unique(groups):
        mask = groups == group
        shuffled[mask] = rng.permutation(y[mask])
    if not any(len(np.unique(y[groups == k])) > 1 for k in np.unique(groups)):
        raise ValueError('Within-batch permutation impossible: every batch has only one sex')
    return shuffled


def compare_files(a,b,out):
    import json
    from pathlib import Path
    import pandas as pd
    frames=[]
    for path in [a,b]:
        df=pd.read_csv(path,dtype={'egg_id':str,'batch_id':str})
        if df.egg_id.duplicated().any():raise ValueError('One prediction per egg is required; select one seed/model first')
        df=df.dropna(subset=['p_female'])
        if not set(df.sex).issubset({'F','M'}):raise ValueError('F/M reference labels required')
        frames.append(df[['egg_id','batch_id','sex','p_female']])
    joined=frames[0].merge(frames[1],on='egg_id',suffixes=('_a','_b'),validate='one_to_one')
    if joined.empty:raise ValueError('No common eggs')
    if not (joined.sex_a.eq(joined.sex_b)&joined.batch_id_a.eq(joined.batch_id_b)).all():
        raise ValueError('Paired labels or batch IDs disagree')
    y=(joined.sex_a=='F').to_numpy(dtype=int)
    result={'common_eggs':len(joined),'valid_a':len(frames[0]),'valid_b':len(frames[1]),
            'a':scores(y,joined.p_female_a),'b':scores(y,joined.p_female_b),
            'accuracy_difference_a_minus_b':cluster_interval(y,joined.p_female_a,joined.batch_id_a,
                                                             other=joined.p_female_b)}
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    Path(out).write_text(json.dumps(result,indent=2,allow_nan=False))
    return result
