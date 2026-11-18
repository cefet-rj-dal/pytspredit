"""Build processed documentation from executable notebooks.

This is the pydaltoolbox counterpart of the R ``Rmd/conversor.R``: it takes the
jupytext-markdown *sources* under ``notebooks/`` (the "Rmd analog"), executes each
one end to end, and writes the rendered Markdown *with the real output embedded*
under ``examples/`` (the "processed documentation" analog).

Because every notebook is executed, a failure here means the example is broken --
so this doubles as a smoke test over the public API.

Usage
-----
    python tools/build_examples.py            # build everything
    python tools/build_examples.py tutorial   # build only notebooks/tutorial/**
    python tools/build_examples.py --check     # fail if any notebook errors
"""
from __future__ import annotations

import sys
from pathlib import Path

import jupytext
from nbconvert import MarkdownExporter
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat.v4 import new_code_cell

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "notebooks"
OUT_DIR = ROOT / "examples"
KERNEL = "python3"
TIMEOUT = 300

# Injected (and then removed) before execution so pandas renders tables as plain
# text without truncation -- keeping the processed output clean, like the R "##"
# style, instead of the verbose HTML tables nbconvert would otherwise emit.
SETUP_CODE = (
    "import pandas as pd\n"
    "pd.set_option('display.max_columns', None)\n"
    "pd.set_option('display.width', 1000)"
)


def _strip_html_outputs(nb) -> None:
    """Drop the text/html representation so Markdown uses the plain-text table."""
    for cell in nb.cells:
        for output in cell.get("outputs", []):
            data = output.get("data")
            if data and "text/html" in data:
                del data["text/html"]


def build_one(src: Path) -> Path:
    """Execute one notebook source and write its processed Markdown."""
    rel = src.relative_to(SRC_DIR)
    out = (OUT_DIR / rel).with_suffix(".md")
    out.parent.mkdir(parents=True, exist_ok=True)

    nb = jupytext.read(src)
    nb.cells.insert(0, new_code_cell(SETUP_CODE))  # hidden setup cell
    ep = ExecutePreprocessor(timeout=TIMEOUT, kernel_name=KERNEL)
    ep.preprocess(nb, {"metadata": {"path": str(src.parent)}})
    nb.cells.pop(0)  # remove setup cell so it never appears in the output
    _strip_html_outputs(nb)

    body, _ = MarkdownExporter().from_notebook_node(nb)
    out.write_text(body, encoding="utf-8")
    return out


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    sub = SRC_DIR / args[0] if args else SRC_DIR
    sources = sorted(sub.rglob("*.md")) if sub.is_dir() else [sub]
    sources = [s for s in sources if s.name != "README.md"]

    if not sources:
        print(f"No notebook sources found under {sub}")
        return 0

    failed = []
    for src in sources:
        try:
            out = build_one(src)
            print(f"OK   {src.relative_to(ROOT)}  ->  {out.relative_to(ROOT)}")
        except Exception as exc:  # noqa: BLE001 - we want to report and continue
            failed.append((src, exc))
            print(f"FAIL {src.relative_to(ROOT)}: {type(exc).__name__}: {exc}")

    if failed:
        print(f"\n{len(failed)} notebook(s) failed.")
        return 1
    print(f"\n{len(sources)} notebook(s) built successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
