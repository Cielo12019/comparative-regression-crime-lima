# Ejecuta limpieza y EDA en Windows PowerShell

python scripts\clean_data.py --input data/raw.csv --output data/cleaned.csv --freq D --n-lags 7
python scripts\eda.py --input data/cleaned.csv --outdir figures
