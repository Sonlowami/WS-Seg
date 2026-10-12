"""
CSV helpers for experiment results: an append-and-flush row writer (so an
interrupted run keeps every finished fit and can --resume), a reader that
parses numeric columns, and a plain writer for summaries.

Interrupted runs can leave three kinds of damage, all handled here:
  * an empty file (killed before the header reached disk)
  * a headerless file (an older RowWriter resumed onto such an empty file)
  * a half-written last line (killed in the middle of writing rows)
"""
import csv
import io
import warnings
from pathlib import Path


def _first_row(path: Path) -> list:
    with open(path, newline="") as f:
        return next(csv.reader(f), [])


def _drop_partial_last_line(path: Path):
    """Truncate an unterminated last line: it is a record cut off mid-write,
    and appending to it would merge two records into one line."""
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        keep = data.rfind(b"\n") + 1
        warnings.warn(f"{path}: dropping an incomplete last line (an interrupted write).")
        path.write_bytes(data[:keep])


def _prepend_header(path: Path, fieldnames: list):
    with open(path, newline="") as f:
        body = f.read()
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(fieldnames)
        f.write(body)


class RowWriter:
    def __init__(self, path: Path, fieldnames: list, resume: bool):
        path = Path(path)
        if path.exists() and not resume:
            raise SystemExit(f"{path} exists; pass --resume to continue it or choose another --out.")
        path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not path.exists() or path.stat().st_size == 0
        if not needs_header:
            _drop_partial_last_line(path)
            header = _first_row(path)
            if not header:
                needs_header = True
            elif header != list(fieldnames):
                if len(header) == len(fieldnames) and not set(header) & set(fieldnames):
                    warnings.warn(f"{path} has rows but no header (written by an interrupted run); "
                                  f"adding the header.")
                    _prepend_header(path, list(fieldnames))
                else:
                    raise SystemExit(
                        f"{path} has columns {header}, but this run writes {list(fieldnames)} "
                        f"(probably an older version's output); choose another --out.")
        self.f = open(path, "a", newline="")
        self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
        if needs_header:
            self.writer.writeheader()
            self.f.flush()             # on disk now, so a crash cannot leave a headerless file

    def write(self, rows: list):
        self.writer.writerows(rows)
        self.f.flush()


def read_rows(paths: list, int_fields=(), float_fields=(), fieldnames: list = None) -> list:
    """
    Rows from one or more CSVs, with the given columns parsed (NaN stays NaN).
    fieldnames: the columns the file was written with; lets a headerless file
    (see module docstring) be read, with a warning. Empty files give no rows;
    incomplete rows from an interrupted write are skipped, with a warning.
    """
    rows = []
    required = set(int_fields) | set(float_fields) | set(fieldnames or ())
    for path in paths:
        path = Path(path)
        header = _first_row(path) if path.stat().st_size else []
        if not header:
            continue
        names = None                                  # the file's own header
        if fieldnames and header != list(fieldnames) and len(header) == len(fieldnames) \
                and not set(header) & set(fieldnames):
            warnings.warn(f"{path} has no header row; reading it with the expected columns.")
            names = list(fieldnames)
        missing = required - set(names or header)
        if missing:
            raise ValueError(f"{path} lacks columns {sorted(missing)}; its first row is {header}. "
                             f"Is it the right file, or written by an older version?")
        with open(path, newline="") as f:
            content = f.read()
        skipped = 0
        if not content.endswith("\n"):
            content = content[:content.rfind("\n") + 1]   # last line cut off mid-write
            skipped += 1
        for r in csv.DictReader(io.StringIO(content), fieldnames=names):
            if None in r or any(v is None for v in r.values()):
                skipped += 1                           # wrong number of fields
                continue
            for key in int_fields:
                r[key] = int(r[key])
            for key in float_fields:
                r[key] = float(r[key])
            rows.append(r)
        if skipped:
            warnings.warn(f"{path}: skipped {skipped} incomplete row(s) from an interrupted write.")
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
