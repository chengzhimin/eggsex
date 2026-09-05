"""Outer batch CV for the three neural modes; final test assets are excluded."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .model import audit
from .benchmark import group_folds
from .deep import prepare,train_deep,predict_deep
from .evaluation import scores,cluster_interval


def run(manifest,config,out,modes=('hsi','rgb','fusion'),seeds=(42,),outer_n=5,
        variant='rgb',epochs=30,batch_size=16,lr=1e-4,size=224,
        pretrained=False,freeze_image=False,patience=7,device='cpu'):
    manifest=Path(manifest).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    config=Path(config).resolve();cfg=json.loads(config.read_text())
    df=pd.read_csv(manifest,dtype=str,keep_default_na=False);audit(df)
    heldout_n=int((df.split=='test').sum());df=df[df.split!='test'].reset_index(drop=True)
    if not modes or not seeds or not set(modes).issubset({'hsi','rgb','fusion'}):
        raise ValueError('Specify at least one valid mode and seed')
    if len(set(modes))!=len(modes) or len(set(seeds))!=len(seeds):
        raise ValueError('Modes and seeds must be distinct')
    # One paired population for all requested modes; report exclusions explicitly.
    qc_mode='fusion' if 'fusion' in modes or {'hsi','rgb'}.issubset(modes) else modes[0]
    df,_,rejected=prepare(df,manifest.parent,cfg,qc_mode,variant,size)
    for column in ['hsi_path','rgb_path','rgb_mask_path']:
        if column in df:
            df[column]=df[column].map(lambda p:str((manifest.parent/p).resolve()) if p else p)
    pd.DataFrame(rejected,columns=['egg_id','reason']).to_csv(out/'qc_rejected.csv',index=False)
    df.to_csv(out/'cohort.csv',index=False)
    y=(df.sex=='F').to_numpy(dtype=int);g=df.batch_id.to_numpy()
    results=[];predictions=[]
    for seed in seeds:
        folds=group_folds(np.zeros((len(y),1)),y,g,outer_n,seed)
        for mode in modes:
            current=[]
            for i,(tr,te) in enumerate(folds):
                # Three disjoint inner partitions: fit, early stopping, calibration.
                inner=group_folds(np.zeros((len(tr),1)),y[tr],g[tr],3,seed+1)
                va=tr[inner[0][1]];ca=tr[inner[1][1]]
                fold_df=df.copy();fold_df['split']='train'
                fold_df.loc[va,'split']='val';fold_df.loc[ca,'split']='calib';fold_df.loc[te,'split']='test'
                fold_out=out/f'{mode}_seed{seed}'/f'fold{i}'
                fold_out.mkdir(parents=True,exist_ok=True)
                fold_df.to_csv(fold_out/'manifest.csv',index=False)
                fold_df.iloc[te].to_csv(fold_out/'outer_test.csv',index=False)
                train_deep(fold_out/'manifest.csv',config,fold_out/'fit',mode=mode,variant=variant,
                           epochs=epochs,batch_size=batch_size,lr=lr,seed=seed,size=size,
                           pretrained=pretrained,freeze_image=freeze_image,patience=patience,device=device)
                predict_deep(fold_out/'fit/model.pt',fold_out/'outer_test.csv',fold_out/'evaluate',evaluate=True,
                             device=device,batch_size=batch_size)
                frame=pd.read_csv(fold_out/'evaluate/predictions.csv',dtype={'egg_id':str,'batch_id':str})
                frame['mode']=mode;frame['seed']=seed;frame['fold']=i
                current.append(frame)
            rec=pd.concat(current,ignore_index=True)
            if rec.egg_id.duplicated().any() or rec.p_female.isna().any() or len(rec)!=len(df):
                raise ValueError('Incomplete or duplicate outer-fold predictions')
            yy=(rec.sex=='F').to_numpy(dtype=int)
            report=scores(yy,rec.p_female)
            report.update(mode=mode,seed=seed,variant=variant,
                          accuracy_batch_ci=cluster_interval(yy,rec.p_female,rec.batch_id.to_numpy(),seed=seed))
            results.append(report);predictions.append(rec)
    pd.concat(predictions).to_csv(out/'oof_predictions.csv',index=False)
    result={'evaluation':'deep outer batch CV with inner early stopping and calibration',
            'final_test_eggs_not_opened':heldout_n,'common_cohort_n':len(df),'synthetic':bool(cfg.get('synthetic',False)),
            'results':results,'note':'Architecture/learning-rate choices are fixed, not searched within these folds. Comparing settings is exploratory; reserve final test.'}
    (out/'deep_cv.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return result
