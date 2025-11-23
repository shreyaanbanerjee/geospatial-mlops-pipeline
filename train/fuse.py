
import argparse, torch, os

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='inp', required=True)
    p.add_argument('--out', dest='out', required=True)
    args = p.parse_args()
    state = torch.load(args.inp, map_location='cpu')
    # demo "fusion": just save a copy (real fusion would fold reparam layers)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(state, args.out)
    print('[OK] fused model written to', args.out)

if __name__=='__main__':
    main()
