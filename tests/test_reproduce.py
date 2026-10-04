"""The committed results/ are exactly what the records produce.

Regenerates the run tables, every LaTeX table and figure of the paper, and the
claims check from records/ into a temporary directory and compares them with the
committed files, byte for byte. ``make results`` rewrites results/ in place.
"""

import contextlib
import filecmp
import io
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

RECORDS = os.path.join(ROOT, "records")
RESULTS = os.path.join(ROOT, "results")


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory):
    import aggregate
    import claims
    import make_report
    out = tmp_path_factory.mktemp("results")
    with contextlib.redirect_stdout(io.StringIO()):
        assert aggregate.main(["--records", RECORDS, "--out", str(out)]) == 0
        assert make_report.main(["--records", RECORDS, "--results", str(out)]) == 0
    text = io.StringIO()
    with contextlib.redirect_stdout(text):
        assert claims.main(["--records", RECORDS, "--results", str(out)]) == 0
    (out / "claims.txt").write_text(text.getvalue().replace(RECORDS, "records"))
    return str(out)


def _files(root):
    return sorted(os.path.relpath(os.path.join(d, f), root)
                  for d, _, files in os.walk(root) for f in files)


def test_same_files(regenerated):
    assert _files(regenerated) == _files(RESULTS)


@pytest.mark.parametrize("name", ["runs.csv", "dense.csv", "claims.txt"])
def test_tables_and_claims_match(regenerated, name):
    assert filecmp.cmp(os.path.join(regenerated, name), os.path.join(RESULTS, name), shallow=False)


def test_paper_tables_and_figures_match(regenerated):
    for sub in ("tables", "figures"):
        names = sorted(os.listdir(os.path.join(RESULTS, sub)))
        _, mismatch, errors = filecmp.cmpfiles(os.path.join(regenerated, sub),
                                               os.path.join(RESULTS, sub), names, shallow=False)
        assert not mismatch and not errors, (sub, mismatch, errors)
