"""
CSV helpers for experiment results: an append-and-flush row writer (so an
interrupted run keeps every finished fit and can --resume), a reader that
parses numeric columns, and a plain writer for summaries.
"""
import csv
from pathlib import Path


class RowWriter:
    def __init__(self, path: Path, fieldnames: list, resume: bool):
        path = Path(path)
        if path.exists() and not resume:
            raise SystemExit(f"{path} exists; pass --resume to continue it or choose another --out.")
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        self.f = open(path, "a", newline="")
        self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
        if new:
            self.writer.writeheader()

    def write(self, rows: list):
        self.writer.writerows(rows)
        self.f.flush()


def read_rows(paths: list, int_fields=(), float_fields=()) -> list:
    """Rows from one or more CSVs, with the given columns parsed (NaN stays NaN)."""
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                for key in int_fields:
                    r[key] = int(r[key])
                for key in float_fields:
                    r[key] = float(r[key])
                rows.append(r)
    return rows


def write_csv(path: Path, rows: list):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
