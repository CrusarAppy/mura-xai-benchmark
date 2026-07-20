"""Append one flat result row (config + metrics) to a CSV results database."""
from __future__ import annotations
from pathlib import Path
from typing import Dict


def append_result(csv_path: str | Path, row: Dict) -> None:
    import pandas as pd
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row])
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(csv_path, index=False)
