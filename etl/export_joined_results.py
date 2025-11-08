"""Utility script that materialises the combined CSV produced by `extract_data`.

The ETL pipeline typically feeds the joined DataFrame straight into Spark for
transformation; however, the big-data ML pipelines introduced in this repo
(Spark MLlib + TensorFlow) expect a persisted CSV that can be shared across
jobs. Running this module writes that joined dataset to disk so the
Spark/TensorFlow scripts can ingest it without relying on Postgres.
"""

from __future__ import annotations

import pathlib

from extract_data import extract_data


def main() -> None:
    output_path = pathlib.Path("data/f1_results_joined.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = extract_data(data_dir="data/")
    df.to_csv(output_path, index=False)
    print(f"Saved joined dataset to {output_path.resolve()}")


if __name__ == "__main__":
    main()
