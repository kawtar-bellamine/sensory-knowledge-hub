"""Convert the SensoMineR .rda files in data/raw to the CSV files used by the project.

Only needed to rebuild the CSVs: `pip install pyreadr`, then `python data/convert_rda.py`.
"""
from pathlib import Path

import pyreadr

HERE = Path(__file__).resolve().parent

for name in ["chocolates", "cocktail"]:
    for table, df in pyreadr.read_r(HERE / "raw" / f"{name}.rda").items():
        if table == "sensopanels":
            continue
        out = HERE / f"{table.replace('.', '_')}.csv"
        df.to_csv(out, index=True, index_label="product" if table != "sensochoc" else "row")
        print(f"{out.name}: {df.shape}")
