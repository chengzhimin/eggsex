import hashlib
import json
from zipfile import BadZipFile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, brier_score_loss, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from .preprocess import extract


def audit(df, labeled=True):
    required = ['egg_id','batch_id','device_id','protocol_id']
    if labeled:
        required += ['sex','split']
    if (not set(required).issubset(df.columns) or df[required].isna().any().any()
            or df[required].astype(str).apply(lambda c: c.str.strip().eq('')).any().any()):
        raise ValueError('Missing metadata columns/values')
    if df['egg_id'].duplicated().any():
        raise ValueError('duplicate egg_id')
    if labeled:
        if not set(df.sex).issubset({'F','M'}):
            raise ValueError('sex must be F or M')
        if set(df.split) != {'train','val','calib','test'}:
            raise ValueError('Require train, val, calib and test partitions')
        if (df.groupby('batch_id')['split'].nunique() > 1).any():
            raise ValueError('Batch leakage across partitions')
        if (df.groupby('split')['sex'].nunique() != 2).any():
            raise ValueError('Each partition needs both labels')


def wilson(k, n):
    if n == 0:
        return [None, None]
    z = 1.95996398454; p = k/n; denom = 1+z*z/n
    center = (p+z*z/(2*n))/denom
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [float(center-half), float(center+half)]


def metrics(y, p, threshold):
    pred = (p >= .5).astype(int)
    take = np.maximum(p, 1-p) >= threshold
    result = {'n': len(y), 'accuracy': float(accuracy_score(y, pred)),
              'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
              'auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
              'brier': float(brier_score_loss(y, p)),
              'confusion_matrix_order_0_1': confusion_matrix(y, pred, labels=[0,1]).tolist(),
              'accuracy_ci95': wilson(int((pred == y).sum()), len(y)),
              'coverage': float(take.mean()), 'accepted_n': int(take.sum()),
              'accepted_accuracy': float((pred[take] == y[take]).mean()) if take.any() else None,
              'accepted_accuracy_ci95': wilson(int(((pred == y)&take).sum()), int(take.sum()))}
    result['by_true_label'] = {}
    for label, name in [(0,'M'),(1,'F')]:
        rows = y == label; accepted = rows & take
        result['by_true_label'][name] = {
            'n': int(rows.sum()), 'accepted_n': int(accepted.sum()),
            'coverage': float(accepted.sum()/rows.sum()) if rows.any() else None,
            'correct_among_accepted_ci95': wilson(int(((pred == y)&accepted).sum()), int(accepted.sum()))}
    return result


def select_threshold(y, p, target, min_per_label):
    for t in np.r_[.5, np.arange(.55, 1, .025), .99, .995]:
        m = metrics(y, p, float(t))
        if all(v['accepted_n'] >= min_per_label and
               v['correct_among_accepted_ci95'][0] >= target for v in m['by_true_label'].values()):
            return float(t)
    return 1.1


def train(manifest, config, out, mode):
    manifest, out = Path(manifest).resolve(), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(config).read_text())
    df = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    audit(df)
    features, retained, rejected, names = [], [], [], None
    for i, row in df.iterrows():
        try:
            f, n, _ = extract(row, manifest.parent, cfg, mode)
            if names is not None and n != names:
                raise ValueError('Feature schema changed')
            features.append(f); retained.append(i); names = n
        except (ValueError, OSError, KeyError, BadZipFile) as exc:
            rejected.append({'egg_id':row.egg_id, 'split':row.split, 'reason':str(exc)})
    pd.DataFrame(rejected, columns=['egg_id','split','reason']).to_csv(out/'qc_rejected.csv', index=False)
    used = df.loc[retained].copy()
    audit(used)
    x = np.stack(features); y = (used.sex == 'F').to_numpy(dtype=int)
    partitions = {s:(used.split == s).to_numpy() for s in ['train','val','calib','test']}
    tr, va, ca, te = [partitions[s] for s in ['train','val','calib','test']]
    best, score, trials = None, -np.inf, []
    for c in [.1, 1., 10.]:
        model = make_pipeline(StandardScaler(), SVC(C=c, kernel='rbf', class_weight='balanced'))
        model.fit(x[tr], y[tr])
        value = roc_auc_score(y[va], model.decision_function(x[va]))
        trials.append({'C':c, 'validation_auc':float(value)})
        if value > score:
            best, score = model, value
    calibrated = CalibratedClassifierCV(FrozenEstimator(best), method='sigmoid').fit(x[ca], y[ca])
    threshold = select_threshold(y[va], calibrated.predict_proba(x[va])[:,1],
                                 cfg['validation_target_lower_bound'], cfg['min_accepted_per_sex'])
    artifact = {'model':calibrated, 'config':cfg, 'mode':mode, 'feature_names':names,
                'threshold':threshold, 'version':'0.1.0', 'production_ready':False,
                'label_map':{'M':0,'F':1}, 'synthetic':bool(cfg.get('synthetic', False)),
                'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}
    joblib.dump(artifact, out/'model.joblib')
    report = {'status':'RESEARCH_ONLY', 'synthetic':artifact['synthetic'], 'mode':mode,
              'threshold':threshold, 'trials':trials, 'qc_rejected_n':len(rejected),
              'note':'Validation selects the model and threshold; test is evaluation only.'}
    for split, rows in partitions.items():
        p = calibrated.predict_proba(x[rows])[:,1]
        report[split] = metrics(y[rows], p, threshold)
        report[split]['input_n'] = int((df.split == split).sum())
        report[split]['end_to_end_coverage'] = report[split]['accepted_n']/report[split]['input_n']
        pred = used.loc[rows,['egg_id','batch_id','sex']].copy()
        pred['p_female'] = p
        pred['decision'] = np.where(np.maximum(p,1-p) < threshold, 'REVIEW', np.where(p>=.5,'F','M'))
        pred.to_csv(out/f'{split}_predictions.csv', index=False)
    (out/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    used.to_csv(out/'used_manifest.csv', index=False)
    (out/'config.json').write_text(json.dumps(cfg, indent=2))
    return report


def predict(model_path, manifest, out):
    artifact = joblib.load(model_path)
    manifest = Path(manifest).resolve()
    df = pd.read_csv(manifest, dtype=str, keep_default_na=False); audit(df, labeled=False)
    result = []
    for _, row in df.iterrows():
        r = {'egg_id':row.egg_id, 'decision':'REVIEW', 'p_female':None,
             'production_ready':False, 'model_version':artifact['version']}
        try:
            f, names, _ = extract(row, manifest.parent, artifact['config'], artifact['mode'])
            if names != artifact['feature_names']:
                raise ValueError('Feature contract mismatch')
            p = float(artifact['model'].predict_proba(f[None,:])[0,1])
            r['p_female'] = p
            if max(p,1-p) >= artifact['threshold']:
                r['decision'] = 'F' if p >= .5 else 'M'
                r['reason'] = 'research_prediction'
            else:
                r['reason'] = 'uncertain_or_validation_gate_failed'
        except (ValueError, OSError, KeyError, BadZipFile) as exc:
            r['reason'] = str(exc)
        result.append(r)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result).to_csv(out, index=False)
    return result
