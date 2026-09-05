import argparse
import json
from pathlib import Path
from .model import train, predict


def main():
    parser = argparse.ArgumentParser(description='Egg-level HSI/RGB research models and grouped evaluation')
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--manifest', required=True); t.add_argument('--config', required=True)
    t.add_argument('--out', required=True); t.add_argument('--mode', choices=['hsi','rgb','fusion'], default='hsi')
    p = sub.add_parser('predict')
    p.add_argument('--model', required=True); p.add_argument('--manifest', required=True); p.add_argument('--out', required=True)
    d = sub.add_parser('demo'); d.add_argument('--out', required=True)
    v = sub.add_parser('preview'); v.add_argument('--image', required=True); v.add_argument('--mask', required=True)
    v.add_argument('--config', required=True); v.add_argument('--out', required=True)
    b = sub.add_parser('benchmark', help='Nested batch CV on development data only')
    b.add_argument('--manifest', required=True); b.add_argument('--config', required=True); b.add_argument('--out', required=True)
    b.add_argument('--modes', default='hsi,rgb,fusion'); b.add_argument('--models', default='logreg,plsda,svm,extra_trees')
    b.add_argument('--seeds', default='42,123,2026'); b.add_argument('--outer-folds', type=int, default=5)
    b.add_argument('--inner-folds', type=int, default=3); b.add_argument('--permutations', type=int, default=0)
    dt = sub.add_parser('deep-train', help='Spectral CNN / ResNet18 / dual branch')
    dt.add_argument('--manifest', required=True); dt.add_argument('--config', required=True); dt.add_argument('--out', required=True)
    dt.add_argument('--mode', choices=['hsi','rgb','fusion'], default='fusion')
    dt.add_argument('--variant', choices=['rgb','green','clahe'], default='rgb')
    dt.add_argument('--epochs', type=int, default=30); dt.add_argument('--batch-size', type=int, default=16)
    dt.add_argument('--lr', type=float, default=1e-4); dt.add_argument('--seed', type=int, default=42)
    dt.add_argument('--size', type=int, default=224); dt.add_argument('--patience', type=int, default=7)
    dt.add_argument('--pretrained', action='store_true'); dt.add_argument('--freeze-image', action='store_true')
    dt.add_argument('--device', default='cpu')
    for command in ['deep-predict','deep-evaluate']:
        dp = sub.add_parser(command)
        dp.add_argument('--checkpoint', required=True); dp.add_argument('--manifest', required=True); dp.add_argument('--out', required=True)
        dp.add_argument('--device', default='cpu'); dp.add_argument('--batch-size', type=int, default=16)
    cp = sub.add_parser('compare', help='Paired fixed-prediction batch bootstrap')
    cp.add_argument('--a', required=True); cp.add_argument('--b', required=True); cp.add_argument('--out', required=True)
    dc = sub.add_parser('deep-cv', help='Grouped outer CV for neural models; final test excluded')
    dc.add_argument('--manifest',required=True); dc.add_argument('--config',required=True); dc.add_argument('--out',required=True)
    dc.add_argument('--modes',default='hsi,rgb,fusion'); dc.add_argument('--seeds',default='42')
    dc.add_argument('--outer-folds',type=int,default=5); dc.add_argument('--variant',choices=['rgb','green','clahe'],default='rgb')
    dc.add_argument('--epochs',type=int,default=30); dc.add_argument('--batch-size',type=int,default=16)
    dc.add_argument('--lr',type=float,default=1e-4); dc.add_argument('--size',type=int,default=224)
    dc.add_argument('--pretrained',action='store_true'); dc.add_argument('--freeze-image',action='store_true')
    dc.add_argument('--patience',type=int,default=7); dc.add_argument('--device',default='cpu')
    args = parser.parse_args()
    if args.command == 'train':
        report = train(args.manifest, args.config, args.out, args.mode)
        print(json.dumps({'status':report['status'], 'synthetic':report['synthetic'], 'test':report['test']}, indent=2))
    elif args.command == 'predict':
        predict(args.model, args.manifest, args.out); print('Research predictions written')
    elif args.command == 'preview':
        from .preprocess import image_features
        image_features(args.image, args.mask, json.loads(Path(args.config).read_text()), args.out)
    elif args.command == 'benchmark':
        from .benchmark import run
        result = run(args.manifest,args.config,args.out,modes=args.modes.split(','),models=args.models.split(','),
                     seeds=tuple(int(s) for s in args.seeds.split(',')),outer_n=args.outer_folds,
                     inner_n=args.inner_folds,permutations=args.permutations)
        print(json.dumps({'cohort_n':result['common_cohort_n'],'experiments':len(result['results']), 'test_evaluated':False}))
    elif args.command == 'deep-train':
        from .deep import train_deep
        kwargs = vars(args).copy(); kwargs.pop('command')
        print(json.dumps(train_deep(**kwargs),indent=2))
    elif args.command in ('deep-predict','deep-evaluate'):
        from .deep import predict_deep
        print(json.dumps(predict_deep(args.checkpoint,args.manifest,args.out,evaluate=args.command=='deep-evaluate',
                                     device=args.device,batch_size=args.batch_size),indent=2))
    elif args.command == 'compare':
        from .evaluation import compare_files
        compare_files(args.a,args.b,args.out)
    elif args.command == 'deep-cv':
        from .deep_cv import run
        kwargs=vars(args).copy();kwargs.pop('command')
        kwargs['modes']=args.modes.split(',');kwargs['seeds']=tuple(int(s) for s in args.seeds.split(','))
        kwargs['outer_n']=kwargs.pop('outer_folds')
        result=run(**kwargs)
        print(json.dumps({'cohort_n':result['common_cohort_n'],'experiments':len(result['results']),'final_test_evaluated':False}))
    else:
        from .demo import run
        run(Path(args.out))


if __name__ == '__main__':
    main()
