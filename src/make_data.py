import argparse
import pandas as pd

def main(in_path, out_path):
    df = pd.read_csv(in_path)
    df.to_csv(out_path, index=False)
    print(f"✅ leído de {in_path} y guardado en {out_path} con shape={df.shape}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    args = ap.parse_args()
    main(args.in_path, args.out_path)
