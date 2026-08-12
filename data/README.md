Coloque aquí sus archivos de datos crudos.

Formato esperado (CSV): una fila por incidente con al menos las siguientes columnas:

- `incident_id` : identificador único (opcional)
- `date` : fecha y hora del incidente en formato compatible con `pandas.to_datetime`, por ejemplo `YYYY-MM-DD HH:MM:SS`
- `category` : tipo de delito (opcional)

El script `clean_data.py` leerá `data/raw.csv` y generará `data/cleaned.csv`.
