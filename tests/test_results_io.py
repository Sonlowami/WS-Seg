import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

from utils.results_io import RowWriter, read_rows

FIELDS = ["case_id", "steps", "dice"]
ROWS = [{"case_id": "c1", "steps": 500, "dice": 0.8}, {"case_id": "c2", "steps": 500, "dice": 0.7}]


def _read(path):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rows = read_rows([path], int_fields=("steps",), float_fields=("dice",), fieldnames=FIELDS)
    return rows, [str(x.message) for x in w]


def test_crash_before_first_write_then_resume():
    """The header must already be on disk when the writer is created, so a
    run killed before its first fit resumes into a readable file."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "r.csv"
        code = (f"import os, sys; sys.path.insert(0, {str(Path.cwd())!r})\n"
                f"from utils.results_io import RowWriter\n"
                f"w = RowWriter({str(path)!r}, {FIELDS!r}, resume=False)\n"
                f"os._exit(1)\n")
        subprocess.run([sys.executable, "-c", code])
        assert path.read_text().strip() == ",".join(FIELDS)
        RowWriter(path, FIELDS, resume=True).write(ROWS)
        rows, warns = _read(path)
        assert [r["case_id"] for r in rows] == ["c1", "c2"] and rows[0]["steps"] == 500 and not warns


def test_empty_file_is_treated_as_new():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "r.csv"
        path.touch()
        assert _read(path)[0] == []
        RowWriter(path, FIELDS, resume=True).write(ROWS)
        assert len(_read(path)[0]) == 2


def test_headerless_file_is_read_and_repaired():
    """Output of the old bug: rows appended to an empty file, no header."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "r.csv"
        path.write_text("c1,500,0.8\nc2,500,0.7\n")
        rows, warns = _read(path)
        assert [r["dice"] for r in rows] == [0.8, 0.7] and any("no header" in w for w in warns)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            RowWriter(path, FIELDS, resume=True).write([{"case_id": "c3", "steps": 500, "dice": 0.6}])
        assert path.read_text().splitlines()[0] == ",".join(FIELDS)
        rows, warns = _read(path)
        assert len(rows) == 3 and not warns


def test_partial_last_line_is_dropped():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "r.csv"
        path.write_text("case_id,steps,dice\nc1,500,0.8\nc2,500,0.7\nc3,500,0.88")   # cut mid-number
        rows, warns = _read(path)
        assert [r["case_id"] for r in rows] == ["c1", "c2"] and any("incomplete" in w for w in warns)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            RowWriter(path, FIELDS, resume=True).write([{"case_id": "c3", "steps": 500, "dice": 0.6}])
        rows, _ = _read(path)
        assert [(r["case_id"], r["dice"]) for r in rows] == [("c1", 0.8), ("c2", 0.7), ("c3", 0.6)]


def test_wrong_columns_fail_clearly():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "r.csv"
        path.write_text("case_id,steps,old_metric\nc1,500,0.8\n")
        try:
            RowWriter(path, FIELDS, resume=True)
        except SystemExit as e:
            assert "older version" in str(e)
        else:
            raise AssertionError("appending to a file with other columns must fail")
        try:
            _read(path)
        except ValueError as e:
            assert "lacks columns ['dice']" in str(e)
        else:
            raise AssertionError("reading a file without required columns must fail clearly")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("Results I/O tests passed.")
