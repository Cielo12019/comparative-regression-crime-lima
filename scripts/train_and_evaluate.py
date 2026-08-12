"""Entrena varios modelos de regresión en los datos limpios y genera tablas comparativas.

Genera:
- `results/default_params.csv` : R2 en entrenamiento y prueba usando parámetros por defecto.
- `results/tuned_params.csv` : R2 con hiperparámetros ajustados por GridSearchCV.
- `results/comparative_table.md` : tabla combinada en Markdown (escenario 1 y 2).

Uso:
    python scripts\train_and_evaluate.py --input data/cleaned.csv --out results
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


def load_data(path):
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def prepare_features(df):
    # use lag features t-1..t-7 and temporal features
    lag_cols = [f't-{i}' for i in range(1,8) if f't-{i}' in df.columns]
    temporal = [c for c in ['weekday','month','weekofyear','hour'] if c in df.columns]
    features = lag_cols + temporal
    X = df[features].copy()
    y = df['count'].copy()
    # fill any remaining NaNs using forward-fill then zeros
    X = X.ffill().fillna(0)
    return X, y


def time_split(X, y, frac=0.8):
    n = int(len(X) * frac)
    X_train, X_test = X.iloc[:n], X.iloc[n:]
    y_train, y_test = y.iloc[:n], y.iloc[n:]
    return X_train, X_test, y_train, y_test


def evaluate_models_with_cv(models, X_train, y_train, X_test, y_test, cv_splits=5):
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    rows = []
    for name, model in models.items():
        # compute CV score on training set
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_train):
            m = model
            m.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            cv_scores.append(r2_score(y_train.iloc[val_idx], m.predict(X_train.iloc[val_idx])))
        mean_cv = np.mean(cv_scores)
        # fit on full training and evaluate on test
        model.fit(X_train, y_train)
        r2_te = r2_score(y_test, model.predict(X_test))
        rows.append({'Model': name, 'Training data': round(mean_cv, 2), 'Test data': round(r2_te, 2)})
    return pd.DataFrame(rows)


def tune_and_evaluate(model_defs, param_grids, X_train, y_train, X_test, y_test, cv_splits=3):
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    rows = []
    for name, estimator in model_defs.items():
        grid = param_grids.get(name, None)
        if grid is None:
            best = estimator.fit(X_train, y_train)
            mean_cv = None
        else:
            gs = GridSearchCV(estimator, grid, cv=tscv, scoring='r2', n_jobs=-1)
            gs.fit(X_train, y_train)
            best = gs.best_estimator_
            mean_cv = gs.best_score_
        # compute test score
        r2_te = r2_score(y_test, best.predict(X_test))
        # training score (use mean_cv if available, else score on training set)
        if mean_cv is None:
            r2_tr = r2_score(y_train, best.predict(X_train))
        else:
            r2_tr = mean_cv
        rows.append({'Model': name, 'Training data': round(r2_tr, 2), 'Test data': round(r2_te, 2)})
    return pd.DataFrame(rows)


def main(args):
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = load_data(input_path)
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = time_split(X, y, frac=args.train_frac)

    # Default models
    default_models = {
        'XGBoost Regression': XGBRegressor(objective='reg:squarederror', n_jobs=1),
        'Extra Tree Regression': ExtraTreesRegressor(n_jobs=-1),
        'Support Vector Regression': SVR(),
        'Bagging Regression': BaggingRegressor(),
        'Random Forest Regression': RandomForestRegressor(n_jobs=-1),
        'AdaBoost Regression': AdaBoostRegressor()
    }

    # Scenario 1: default models evaluated with TimeSeriesSplit CV on training set
    results_default = evaluate_models_with_cv(default_models, X_train, y_train, X_test, y_test, cv_splits=5)
    results_default.to_csv(outdir / 'default_params.csv', index=False)

    # Parameter grids (small grids for speed)
    param_grids = {
        'XGBoost Regression': {'n_estimators': [50, 100], 'max_depth': [3, 6]},
        'Extra Tree Regression': {'n_estimators': [50, 100], 'max_depth': [None, 10]},
        'Support Vector Regression': {'C': [1, 10], 'gamma': ['scale']},
        'Bagging Regression': {'n_estimators': [10, 50]},
        'Random Forest Regression': {'n_estimators': [50, 100], 'max_depth': [None, 10]},
        'AdaBoost Regression': {'n_estimators': [50, 100], 'learning_rate': [0.5, 1.0]}
    }

    results_tuned = tune_and_evaluate(default_models, param_grids, X_train, y_train, X_test, y_test, cv_splits=3)
    results_tuned.to_csv(outdir / 'tuned_params.csv', index=False)

    # Combined comparative markdown
    # Save comparative table and LaTeX versions
    combined = results_default.merge(results_tuned, on='Model', suffixes=('_default','_tuned'))
    combined = combined[['Model','Training data_default','Test data_default','Training data_tuned','Test data_tuned']]
    combined.columns = ['Model','Train R2 (default)','Test R2 (default)','Train R2 (tuned)','Test R2 (tuned)']
    combined.to_csv(outdir / 'comparative_table.csv', index=False)
    combined.to_markdown(outdir / 'comparative_table.md', index=False)
    # LaTeX table
    tex = combined.to_latex(index=False,float_format="%.2f")
    (outdir / 'comparative_table.tex').write_text(tex)
    print('Results written to', outdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/cleaned.csv')
    parser.add_argument('--outdir', default='results')
    parser.add_argument('--train-frac', type=float, default=0.8)
    args = parser.parse_args()
    main(args)
