import argparse
import json
from pathlib import Path
from .model import train, predict


def main():
    parser = argparse.ArgumentParser(description='Research baseline; not a validated production product')
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--manifest', required=True); t.add_argument('--config', required=True)
    t.add_argument('--out', required=True); t.add_argument('--mode', choices=['hsi','rgb','fusion'], default='hsi')
    p = sub.add_parser('predict')
    p.add_argument('--model', required=True); p.add_argument('--manifest', required=True); p.add_argument('--out', required=True)
    d = sub.add_parser('demo'); d.add_argument('--out', required=True)
    v = sub.add_parser('preview'); v.add_argument('--image', required=True); v.add_argument('--mask', required=True)
    v.add_argument('--config', required=True); v.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.command == 'train':
        report = train(args.manifest, args.config, args.out, args.mode)
        print(json.dumps({'status':report['status'], 'synthetic':report['synthetic'], 'test':report['test']}, indent=2))
    elif args.command == 'predict':
        predict(args.model, args.manifest, args.out); print('Research predictions written')
    elif args.command == 'preview':
        from .preprocess import rgb_features
        rgb_features(args.image, args.mask, json.loads(Path(args.config).read_text()), args.out)
    else:
        from .demo import run
        run(Path(args.out))


if __name__ == '__main__':
    main()
