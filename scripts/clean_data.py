"""Limpiar datos y generar features para modelado.

Uso:
    python scripts\clean_data.py --input data/raw.csv --output data/cleaned.csv

Genera columnas: `year`, `month`, `day`, `hour`, `weekday`, `weekofyear`, `count` (agregada por periodo si se especifica), y columnas de ventana deslizante `t-7`..`t-1`.
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def load_data(path):
    # Try to read semicolon-delimited files (common in local exports)
    try:
        df = pd.read_csv(path, sep=';', low_memory=False)
        return df
    except Exception:
        df = pd.read_csv(path)
        return df


def basic_cleanup(df, datetime_col='date'):
    df = df.copy()
    # If a datetime column already exists, try to parse it
    if datetime_col in df.columns:
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')

    # If not parsed or not present, try to build datetime from known fields in the dataset
    if datetime_col not in df.columns or df[datetime_col].isna().all():
        # Expected fields in your provided CSV: IH203_DIA, IH203_MES, IH203_ANIO, IH204_HOR, IH204_MIN
        if all(col in df.columns for col in ['IH203_ANIO', 'IH203_MES', 'IH203_DIA']):
            # coerce to numeric and handle invalid markers like 99 or 'SN'
            y = pd.to_numeric(df['IH203_ANIO'], errors='coerce').fillna(0).astype(int)
            m = pd.to_numeric(df['IH203_MES'], errors='coerce').fillna(1).astype(int)
            d = pd.to_numeric(df['IH203_DIA'], errors='coerce').fillna(1).astype(int)
            # hour/min may have 99 or missing -> set to 0
            if 'IH204_HOR' in df.columns:
                hh = pd.to_numeric(df['IH204_HOR'], errors='coerce')
                hh = hh.where((hh >= 0) & (hh < 24), 0).fillna(0).astype(int)
            else:
                hh = 0
            if 'IH204_MIN' in df.columns:
                mm = pd.to_numeric(df['IH204_MIN'], errors='coerce')
                mm = mm.where((mm >= 0) & (mm < 60), 0).fillna(0).astype(int)
            else:
                mm = 0
            # Construct datetime
            dates = pd.to_datetime(dict(year=y, month=m, day=d, hour=hh, minute=mm), errors='coerce')
            df[datetime_col] = dates

    df = df.dropna(subset=[datetime_col])
    df['year'] = df[datetime_col].dt.year
    df['month'] = df[datetime_col].dt.month
    df['day'] = df[datetime_col].dt.day
    df['hour'] = df[datetime_col].dt.hour
    df['weekday'] = df[datetime_col].dt.weekday
    df['weekofyear'] = df[datetime_col].dt.isocalendar().week.astype(int)
    return df


def aggregate_counts(df, freq='D', datetime_col='date'):
    df = df.copy()
    df.set_index(pd.to_datetime(df[datetime_col]), inplace=True)
    counts = df.resample(freq).size().rename('count').to_frame()
    counts.reset_index(inplace=True)
    counts.rename(columns={datetime_col: 'date'}, inplace=True)
    counts['year'] = counts['date'].dt.year
    counts['day'] = counts['date'].dt.day
    counts['hour'] = counts['date'].dt.hour
    counts['weekday'] = counts['date'].dt.weekday
    counts['weekofyear'] = counts['date'].dt.isocalendar().week.astype(int)
    return counts


def sliding_window(df_counts, target_col='count', n_lags=7):
    df = df_counts.copy()
    for lag in range(1, n_lags+1):
        df[f't-{lag}'] = df[target_col].shift(lag)
    df = df.dropna().reset_index(drop=True)
    return df


def main(args):
    input_path = Path(args.input)
    output_path = Path(args.output)
    df = load_data(input_path)
    df = basic_cleanup(df, datetime_col=args.datetime_col)
    # aggregate to daily counts by default
    counts = aggregate_counts(df, freq=args.freq, datetime_col=args.datetime_col)
    counts_sw = sliding_window(counts, target_col='count', n_lags=args.n_lags)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts_sw.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/raw.csv')
    parser.add_argument('--output', default='data/cleaned.csv')
    parser.add_argument('--datetime-col', default='date')
    parser.add_argument('--freq', default='D', help='resample frequency e.g. D, H, W')
    parser.add_argument('--n-lags', type=int, default=7)
    args = parser.parse_args()
    main(args)
