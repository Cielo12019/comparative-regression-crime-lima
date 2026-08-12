# Proyecto: Análisis y limpieza de datos de delitos (Lima)

_Este repositorio contiene scripts para limpiar datos, generar features para ventanas deslizantes y producir análisis exploratorio (EDA) con gráficas similares a las imágenes proporcionadas._

- `data/` : carpeta para datos de entrada y salida.
- `scripts/clean_data.py` : script para limpieza y generación de features.
- `scripts/eda.py` : script para generar gráficas y tablas EDA.
- `run_all.ps1` : script para ejecutar todo en Windows PowerShell.

Instrucciones rápidas:

1. Crear un entorno virtual e instalar dependencias:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Colocar el CSV original en `data/raw.csv` (ver `data/README.md` para formato esperado).
3. Ejecutar limpieza y EDA:

```powershell
.\run_all.ps1
```

Opcional: generar datos de ejemplo para probar el flujo:

```powershell
python scripts\generate_sample.py --start-year 2015 --end-year 2017 --out data/raw.csv
```

# Thesis_ML
