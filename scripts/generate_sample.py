"""Generador de datos de ejemplo para probar el pipeline.

Genera un CSV con registros horarios simulados para varios años.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def generate(start_year=2015, end_year=2017, outfile='data/raw.csv'):
    rng = pd.date_range(start=f'{start_year}-01-01', end=f'{end_year}-12-31 23:00:00', freq='H')
    # simulate counts per hour as Poisson with daily pattern
    hours = rng.hour
    base = 20 + (hours >= 18) * 30 + (hours <= 4) * 15  # more events in evenings
    counts = np.random.poisson(lam=base)
    rows = []
    for ts, c in zip(rng, counts):
        for i in range(c):
            rows.append({'date': ts, 'category': np.random.choice(['theft','assault','robbery'], p=[0.6,0.25,0.15])})
    df = pd.DataFrame(rows)
    outp = Path(outfile)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False)
    print(f'Sample data saved to {outp} ({len(df)} rows)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-year', type=int, default=2015)
    parser.add_argument('--end-year', type=int, default=2017)
    parser.add_argument('--out', default='data/raw.csv')
    args = parser.parse_args()
    generate(args.start_year, args.end_year, args.out)
