"""Trainable spectral CNN / ResNet18 / dual-branch model for egg-level research.

No pretrained sex classifier is bundled. ImageNet initialization is optional.
"""
import hashlib
import json
import os
import random
from functools import lru_cache
from pathlib import Path
from zipfile import BadZipFile
import cv2
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from .model import audit
from .preprocess import cube_features, image_features, check_context
from .evaluation import scores, cluster_interval


class SpectralEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(3,32,5,padding=2),nn.GroupNorm(8,32),nn.SiLU(),
            nn.Conv1d(32,64,5,padding=2),nn.GroupNorm(8,64),nn.SiLU(),
            nn.Conv1d(64,64,3,padding=1),nn.GroupNorm(8,64),nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),nn.Flatten())

    def forward(self,x):
        return self.layers(x)


class EggNet(nn.Module):
    def __init__(self, mode='fusion', pretrained=False, freeze_image=False):
        super().__init__()
        if mode not in ('hsi','rgb','fusion'):
            raise ValueError('Unknown mode')
        self.mode=mode;self.freeze_image=freeze_image
        dim=0
        if mode in ('hsi','fusion'):
            self.spectral=SpectralEncoder();dim+=64
        if mode in ('rgb','fusion'):
            self.image=resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            self.image.fc=nn.Identity();dim+=512
            if freeze_image:
                for p in self.image.parameters():
                    p.requires_grad=False
        self.head=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,64),nn.SiLU(),nn.Dropout(.25),nn.Linear(64,1))

    def train(self,mode=True):
        super().train(mode)
        if self.freeze_image and hasattr(self,'image'):
            self.image.eval()
        return self

    def forward(self,spectrum,image):
        parts=[]
        if self.mode in ('hsi','fusion'):
            parts.append(self.spectral(spectrum))
        if self.mode in ('rgb','fusion'):
            parts.append(self.image(image))
        return self.head(torch.cat(parts,dim=1)).squeeze(-1)


def image_array(row,base,cfg,variant,size):
    image_features(base/row['rgb_path'],base/row['rgb_mask_path'],cfg)
    bgr=cv2.imread(str(base/row['rgb_path']))
    mask=cv2.imread(str(base/row['rgb_mask_path']),cv2.IMREAD_GRAYSCALE)>0
    yy,xx=np.where(mask)
    crop=np.s_[yy.min():yy.max()+1,xx.min():xx.max()+1]
    bgr,mask=bgr[crop].copy(),mask[crop]
    if variant=='rgb':
        arr=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    elif variant in ('green','clahe'):
        arr=bgr[...,1].copy()
        arr[~mask]=int(np.median(arr[mask]))
        if variant=='clahe':
            # Match the network's input scale before local equalization.
            scale=size/max(arr.shape);wh=(max(1,round(arr.shape[1]*scale)),max(1,round(arr.shape[0]*scale)))
            arr=cv2.resize(arr,wh,interpolation=cv2.INTER_AREA)
            mask=cv2.resize(mask.astype('uint8'),wh,interpolation=cv2.INTER_NEAREST)>0
            arr=cv2.createCLAHE(clipLimit=cfg['clahe_clip'],tileGridSize=tuple(cfg['clahe_grid'])).apply(arr)
        arr=np.repeat(arr[...,None],3,axis=2)
    else:
        raise ValueError('image variant must be rgb, green or clahe')
    arr[~mask]=0
    scale=size/max(arr.shape[:2]);wh=(max(1,round(arr.shape[1]*scale)),max(1,round(arr.shape[0]*scale)))
    arr=cv2.resize(arr,wh,interpolation=cv2.INTER_AREA)
    canvas=np.zeros((size,size,3),dtype=np.float32)
    y0=(size-arr.shape[0])//2;x0=(size-arr.shape[1])//2
    canvas[y0:y0+arr.shape[0],x0:x0+arr.shape[1]]=arr/255.
    # Fixed ImageNet normalization; not fitted using held-out eggs.
    return ((canvas-np.array([.485,.456,.406]))/np.array([.229,.224,.225])).transpose(2,0,1).astype('float32')


def prepare(df,base,cfg,mode,variant,size,allow_empty=False):
    keep,features,rejected=[],[],[]
    for i,row in df.iterrows():
        try:
            check_context(row,cfg)
            f=np.zeros((3,len(cfg['wavelengths_nm'])),dtype='float32')
            if mode in ('hsi','fusion'):
                f=cube_features(base/row['hsi_path'],cfg)[0].reshape(3,-1).astype('float32')
            if mode in ('rgb','fusion'):
                image_array(row,base,cfg,variant,size)
            keep.append(i);features.append(f)
        except (OSError,ValueError,KeyError,BadZipFile) as exc:
            rejected.append({'egg_id':row.egg_id,'reason':str(exc)})
    if not keep and not allow_empty:
        raise ValueError('No usable samples after QC')
    array=np.stack(features) if keep else np.empty((0,3,len(cfg['wavelengths_nm'])),dtype='float32')
    return df.loc[keep].reset_index(drop=True),array,rejected


class EggDataset(Dataset):
    def __init__(self,df,spectra,base,cfg,mode,variant,size,augment=False):
        self.df=df.reset_index(drop=True);self.spectra=spectra
        self.base=base;self.cfg=cfg;self.mode=mode;self.variant=variant;self.size=size;self.augment=augment

    def __len__(self):
        return len(self.df)

    @lru_cache(maxsize=16)
    def image(self,i):
        if self.mode=='hsi':
            return torch.zeros(3,1,1)
        return torch.from_numpy(image_array(self.df.iloc[i],self.base,self.cfg,self.variant,self.size))

    def __getitem__(self,i):
        img=self.image(i).clone()
        if self.augment and self.mode!='hsi':
            if torch.rand(())<.5:img=img.flip(-1)
            if torch.rand(())<.5:img=img.flip(-2)
        label=float(self.df.iloc[i].get('sex','M')=='F')
        return torch.from_numpy(self.spectra[i]),img,torch.tensor(label,dtype=torch.float32)


def logits(model,loader,device):
    model.eval();result=[]
    with torch.no_grad():
        for spec,img,_ in loader:
            result.extend(model(spec.to(device),img.to(device)).cpu().tolist())
    return np.asarray(result)


def temperature_fit(logit,y):
    # A one-parameter calibration fit on the dedicated calibration set.
    result=minimize_scalar(lambda t:float(np.mean(np.logaddexp(0,logit/t)-y*logit/t)),
                           bounds=(.05,10.),method='bounded')
    if not result.success:raise ValueError('Temperature calibration failed')
    return float(result.x)


def train_deep(manifest,config,out,mode='fusion',variant='rgb',epochs=30,
               batch_size=16,lr=1e-4,seed=42,size=224,pretrained=False,
               freeze_image=False,patience=7,device='cpu'):
    if epochs<1 or batch_size<1 or size<64 or patience<1 or lr<=0:
        raise ValueError('Invalid training parameters; size must be at least 64')
    if freeze_image and not pretrained:
        raise ValueError('Freezing a random image encoder is not supported; use --pretrained')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    manifest=Path(manifest).resolve();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path(config).read_text());original=pd.read_csv(manifest,dtype=str,keep_default_na=False)
    audit(original)
    dev=original[original.split!='test'].reset_index(drop=True)
    df,x,rejected=prepare(dev,manifest.parent,cfg,mode,variant,size)
    splits={s:np.where(df.split.to_numpy()==s)[0] for s in ['train','val','calib']}
    y=(df.sex=='F').to_numpy(dtype=int)
    if any(len(np.unique(y[ids]))!=2 for ids in splits.values()):
        raise ValueError('Every development partition needs both sexes after QC')
    # Only training eggs determine per-channel, per-band normalization.
    mean=x[splits['train']].mean(axis=0);std=x[splits['train']].std(axis=0)
    std=np.maximum(std,1e-6);x=((x-mean)/std).astype('float32')
    loaders={}
    for name,ids in splits.items():
        ds=EggDataset(df.iloc[ids],x[ids],manifest.parent,cfg,mode,variant,size,augment=name=='train')
        loaders[name]=DataLoader(ds,batch_size=batch_size,shuffle=name=='train',num_workers=0,
                                generator=torch.Generator().manual_seed(seed))
    model=EggNet(mode,pretrained,freeze_image).to(device)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=1e-4)
    loss_fn=nn.BCEWithLogitsLoss()
    best,best_loss,best_epoch,stale=None,np.inf,0,0
    history=[]
    for epoch in range(1,epochs+1):
        model.train();loss_sum=0.;n=0
        for spec,img,target in loaders['train']:
            optimizer.zero_grad(set_to_none=True)
            output=model(spec.to(device),img.to(device));loss=loss_fn(output,target.to(device))
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            loss_sum+=float(loss.detach())*len(target);n+=len(target)
        val_logits=logits(model,loaders['val'],device);vy=y[splits['val']]
        val_loss=float(np.mean(np.logaddexp(0,val_logits)-vy*val_logits))
        history.append({'epoch':epoch,'train_loss':loss_sum/n,'val_loss':val_loss})
        if val_loss<best_loss:
            best_loss=val_loss;best_epoch=epoch;stale=0
            best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        if stale>=patience:break
    model.load_state_dict(best)
    temperature=temperature_fit(logits(model,loaders['calib'],device),y[splits['calib']])
    manifest_hash=hashlib.sha256(manifest.read_bytes()).hexdigest()
    checkpoint={'state_dict':best,'mode':mode,'variant':variant,'size':size,'config':cfg,
                'mean':torch.from_numpy(mean),'std':torch.from_numpy(std),'temperature':temperature,
                'version':'0.2.0','seed':seed,'label_map':{'M':0,'F':1},'freeze_image':freeze_image,
                'pretrained_requested':pretrained,'manifest_sha256':manifest_hash,
                'development_egg_ids':dev.egg_id.tolist(),'development_batch_ids':sorted(set(dev.batch_id)),
                'training_arguments':{'epochs':epochs,'batch_size':batch_size,'lr':lr,'patience':patience}}
    torch.save(checkpoint,out/'model.pt')
    p=expit(logits(model,loaders['val'],device)/temperature)
    report={'mode':mode,'variant':variant,'best_epoch':best_epoch,'temperature':temperature,
            'synthetic':bool(cfg.get('synthetic',False)),'val':scores(y[splits['val']],p),
            'test_evaluated':False,'parameters':sum(p.numel() for p in model.parameters()),
            'note':'Validation selects checkpoint; calibration fits temperature. Test files were not opened.'}
    (out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    pd.DataFrame(history).to_csv(out/'history.csv',index=False)
    pd.DataFrame(rejected,columns=['egg_id','reason']).to_csv(out/'qc_rejected.csv',index=False)
    df.to_csv(out/'used_development_manifest.csv',index=False)
    return report


def predict_deep(checkpoint,manifest,out,evaluate=False,device='cpu',batch_size=16):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=EggNet(ck['mode'],False,ck['freeze_image']).to(device);model.load_state_dict(ck['state_dict'])
    manifest=Path(manifest).resolve();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(manifest,dtype=str,keep_default_na=False)
    audit(df,labeled=False)
    if evaluate:
        if 'sex' not in df or not set(df.sex).issubset({'F','M'}):raise ValueError('F/M labels required for evaluation')
        if set(df.egg_id)&set(ck['development_egg_ids']) or set(df.batch_id)&set(ck['development_batch_ids']):
            raise ValueError('Evaluation overlaps development eggs or batches')
    valid,x,rejected=prepare(df,manifest.parent,ck['config'],ck['mode'],ck['variant'],ck['size'],allow_empty=True)
    x=((x-ck['mean'].numpy())/ck['std'].numpy()).astype('float32')
    ds=EggDataset(valid,x,manifest.parent,ck['config'],ck['mode'],ck['variant'],ck['size'])
    p=expit(logits(model,DataLoader(ds,batch_size=batch_size),device)/ck['temperature'])
    predictions=valid[['egg_id','batch_id']].copy()
    if 'sex' in valid:predictions['sex']=valid.sex
    predictions['p_female']=p;predictions['prediction']=np.where(p>=.5,'F','M');predictions['status']='valid'
    if rejected:
        invalid=pd.DataFrame(rejected);invalid['status']='invalid_input'
        invalid['prediction']='';invalid['p_female']=np.nan
        predictions=pd.concat([predictions,invalid],ignore_index=True)
    predictions.to_csv(out/'predictions.csv',index=False)
    if evaluate:
        y=(valid.sex=='F').to_numpy(dtype=int)
        report=scores(y,p) if len(valid) else {'n':0,'note':'No valid input; no classification metrics'}
        report['input_n']=len(df);report['qc_valid_fraction']=len(valid)/len(df)
        report['accuracy_batch_ci']=cluster_interval(y,p,valid.batch_id.to_numpy()) if len(valid) else None
        report['synthetic']=bool(ck['config'].get('synthetic',False))
        (out/'evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        return report
    return {'input_n':len(df),'valid_n':len(valid)}
