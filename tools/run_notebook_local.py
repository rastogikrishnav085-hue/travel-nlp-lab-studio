"""Run the real notebook on this computer (no Colab) and produce travel_nlp_run_bundle.zip.

Usage (from the project folder, with the virtual environment active):
    python tools/run_notebook_local.py                 # uses every CSV in my_data/
    python tools/run_notebook_local.py --atis          # uses the bundled ATIS train/test files
    python tools/run_notebook_local.py --data some_folder --out some_output_folder

It never modifies your input CSVs: they are copied into a scratch folder (local_run/) first.
The finished bundle is copied to output/travel_nlp_run_bundle.zip (plus a time-stamped copy).
Requires scikit-learn and ipython (pip install -r requirements-dev.txt).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "ATIS_Complete_Pipeline.ipynb"


def notebook_as_script() -> str:
    """Turn the notebook's code cells into one script; Colab-only pip/shell magics are dropped."""
    parts = ["import matplotlib; matplotlib.use('Agg')",
             "try:\n    from IPython.display import display\nexcept ImportError:\n    display = print"]
    for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]:
        if cell["cell_type"] != "code":
            continue
        lines = "".join(cell["source"]).splitlines()
        parts.append("\n".join(("pass  # " + ln) if ln.lstrip().startswith(("!", "%")) else ln for ln in lines))
    return "\n\n".join(parts)


def fail(message: str) -> int:
    print(f"\nERROR: {message}")
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "my_data"), help="folder with your labelled CSV files (default: my_data)")
    ap.add_argument("--atis", action="store_true", help="use the bundled ATIS train/test CSVs instead")
    ap.add_argument("--work", default=str(ROOT / "local_run"), help="scratch folder (deleted and recreated)")
    ap.add_argument("--out", default=str(ROOT / "output"), help="where the finished bundle is copied")
    args = ap.parse_args(argv)

    try:
        import sklearn  # noqa: F401
    except ImportError:
        return fail("scikit-learn is not installed. Run:  pip install -r requirements-dev.txt")

    if args.atis:
        csvs = [ROOT / "data" / "raw" / "train_atis.csv", ROOT / "data" / "raw" / "test_atis.csv"]
        csvs = [p for p in csvs if p.is_file()]
    else:
        data = Path(args.data)
        csvs = sorted(data.glob("*.csv")) if data.is_dir() else []
    if not csvs:
        return fail(f"No CSV files found. Put your labelled CSV files (a text column and an intent column) in "
                    f"'{Path(args.data).name}' and run again, or use --atis for the bundled ATIS sample.")

    work, out = Path(args.work).resolve(), Path(args.out).resolve()
    if ROOT not in work.parents:      # the scratch folder is deleted and recreated, so it must live inside the project
        return fail("The scratch folder (--work) must be a sub-folder of the project. Refusing to delete anything else.")
    shutil.rmtree(work, ignore_errors=True)
    (work / "data" / "raw").mkdir(parents=True)
    for p in csvs:
        shutil.copy2(p, work / "data" / "raw" / p.name)
    print("Input files:", ", ".join(p.name for p in csvs))

    script = work / "run_notebook.py"
    script.write_text(notebook_as_script(), encoding="utf-8")
    log_path = work / "run.log"
    print("Running the notebook (this can take a few minutes)...")
    started = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run([sys.executable, str(script)], cwd=work, stdin=subprocess.DEVNULL,
                              stdout=log, stderr=subprocess.STDOUT, text=True)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if proc.returncode != 0 or "FINAL STATUS: PASS" not in text:
        print("\n".join(text.splitlines()[-25:]))
        return fail(f"The notebook did not finish cleanly. Full log: {log_path}")

    bundle = work / "travel_nlp_run_bundle.zip"
    if not bundle.is_file():
        return fail("The notebook finished but did not create travel_nlp_run_bundle.zip.")
    out.mkdir(parents=True, exist_ok=True)
    final = out / "travel_nlp_run_bundle.zip"
    shutil.copy2(bundle, final)
    shutil.copy2(bundle, out / f"travel_nlp_run_bundle_{time.strftime('%Y%m%d_%H%M%S')}.zip")
    for line in text.splitlines():
        if line.startswith(("Held-out test,", "Fit check:", "FINAL STATUS")):
            print(line)
    print(f"\nDone in {time.time() - started:.0f}s.\nBundle: {final}\nIt now appears in the Studio sidebar under 'Open run' (trained here).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
