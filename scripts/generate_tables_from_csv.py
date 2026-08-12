import pandas as pd
from pathlib import Path

def main(csv_path, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    md = df.to_markdown(index=False)
    (outdir / 'comparative_table.md').write_text(md)
    tex = df.to_latex(index=False, float_format="%.2f")
    (outdir / 'comparative_table.tex').write_text(tex)
    print('Wrote md and tex to', outdir)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='comparative-regression-crime-lima/results/comparative_table.csv')
    parser.add_argument('--out', default='comparative-regression-crime-lima/results')
    args = parser.parse_args()
    main(args.csv, args.out)
