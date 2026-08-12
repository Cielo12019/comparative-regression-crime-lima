"""Generar gráficas EDA similares a las provistas en las imágenes.

Uso:
    python scripts\eda.py --input data/cleaned.csv --outdir figures
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def annotate_bars(ax, fmt="{:.0f}"):
    for p in ax.patches:
        try:
            height = p.get_height()
            if pd.isna(height):
                continue
            ax.annotate(fmt.format(height),
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=8, rotation=90)
        except Exception:
            continue


def plot_counts_by_day(df, outdir):
    # pivot to have grouped bars per day with years as series
    pv = df.groupby(['day', 'year'])['count'].sum().unstack(fill_value=0)
    ax = pv.plot(kind='bar', figsize=(14,6))
    ax.set_title('Count of occurrences per day and year')
    ax.set_xlabel('Day')
    ax.set_ylabel('Count')
    ax.legend(title='Year')
    annotate_bars(ax)
    plt.tight_layout()
    plt.savefig(outdir / 'counts_by_day.png', dpi=200)
    plt.close()


def plot_counts_by_hour(df, outdir):
    pv = df.groupby(['hour', 'year'])['count'].sum().unstack(fill_value=0)
    ax = pv.plot(kind='bar', figsize=(14,6))
    ax.set_title('Count of occurrences per hour and year')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Count')
    ax.legend(title='Year')
    annotate_bars(ax)
    plt.tight_layout()
    plt.savefig(outdir / 'counts_by_hour.png', dpi=200)
    plt.close()


def plot_weekly_series(df, outdir):
    df = df.copy()
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    plt.figure(figsize=(12,8))
    for year in sorted(df['year'].unique()):
        s = df[df['year']==year].groupby('week')['count'].sum()
        plt.plot(s.index, s.values, marker='o', label=str(year))
    plt.title('Occurrence count per week of the year')
    plt.xlabel('Week of the year')
    plt.ylabel('Occurrences')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / 'counts_by_week.png', dpi=200)
    plt.close()


def plot_combined_grid(df, outdir):
    # small multiplot similar to Fig.5 metrics layout (placeholder metrics)
    fig, axes = plt.subplots(2,2, figsize=(12,8))
    sns.barplot(x='year', y='count', data=df.groupby(['year']).sum().reset_index(), ax=axes[0,0])
    axes[0,0].set_title('Total by year')

    sns.barplot(x='month', y='count', data=df.groupby(['month']).sum().reset_index(), ax=axes[0,1])
    axes[0,1].set_title('Total by month')

    # hourly distribution
    pv = df.groupby(['hour','year'])['count'].sum().unstack(fill_value=0)
    pv.plot(kind='bar', ax=axes[1,0], legend=False)
    axes[1,0].set_title('Hourly by year')

    # empty legend panel
    axes[1,1].axis('off')
    axes[1,1].legend(*axes[1,0].get_legend_handles_labels(), loc='center')

    plt.tight_layout()
    plt.savefig(outdir / 'combined_grid.png', dpi=200)
    plt.close()


def main(args):
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    # ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df.get('date')):
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).copy()
    if 'year' not in df.columns:
        df['year'] = df['date'].dt.year
    if 'day' not in df.columns:
        df['day'] = df['date'].dt.day
    if 'hour' not in df.columns:
        df['hour'] = df['date'].dt.hour

    plot_counts_by_day(df, outdir)
    plot_counts_by_hour(df, outdir)
    plot_weekly_series(df, outdir)
    plot_combined_grid(df, outdir)
    print(f"Figures saved to {outdir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/cleaned.csv')
    parser.add_argument('--outdir', default='figures')
    args = parser.parse_args()
    main(args)
