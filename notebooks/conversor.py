"""Convert the Markdown notebook sources into rendered examples.

Executable jupytext Markdown files under ``notebooks/`` are run and exported to
the same relative location under ``examples/``. Binary outputs (plots, for
example) are written to ``fig/<notebook-name>/`` next to the rendered document.
README files are navigation sources and are copied without notebook execution.

Run from any working directory::

    python notebooks/conversor.py
    python notebooks/conversor.py graphics
    python notebooks/conversor.py graphics/comparison/01-comparison-bar.md
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

import jupytext
from nbconvert import MarkdownExporter
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat.v4 import new_code_cell

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "notebooks"
OUT_DIR = ROOT / "examples"
KERNEL = "python3"
TIMEOUT = 300

SETUP_CODE = (
    "import pandas as pd\n"
    "pd.set_option('display.max_columns', None)\n"
    "pd.set_option('display.width', 1000)"
)

MARKDOWN_LINK = re.compile(r"!?\[[^]]*]\(([^)]+)\)")


def _strip_html_outputs(nb) -> None:
    """Prefer stable plain-text tables over verbose HTML representations."""
    for cell in nb.cells:
        for output in cell.get("outputs", []):
            data = output.get("data")
            if data and "text/html" in data:
                del data["text/html"]


def _safe_output_path(document_dir: Path, resource_name: str) -> Path:
    """Resolve an nbconvert resource without allowing it outside examples/."""
    normalized = resource_name.replace("\\", "/")
    target = (document_dir / Path(*PurePosixPath(normalized).parts)).resolve()
    examples_root = OUT_DIR.resolve()
    if target != examples_root and examples_root not in target.parents:
        raise ValueError(f"Unsafe generated resource path: {resource_name}")
    return target


def _write_rendered_notebook(src: Path, out: Path) -> None:
    nb = jupytext.read(src)
    nb.cells.insert(0, new_code_cell(SETUP_CODE))
    ExecutePreprocessor(timeout=TIMEOUT, kernel_name=KERNEL).preprocess(
        nb, {"metadata": {"path": str(src.parent)}}
    )
    nb.cells.pop(0)
    _strip_html_outputs(nb)

    figure_rel = PurePosixPath("fig", src.stem).as_posix()
    body, resources = MarkdownExporter().from_notebook_node(
        nb, resources={"output_files_dir": figure_rel}
    )
    figure_dir = out.parent / "fig" / src.stem
    if figure_dir.exists():
        shutil.rmtree(figure_dir)

    for name, data in resources.get("outputs", {}).items():
        # nbconvert uses os.path.join internally. Normalize only generated
        # resource references, leaving backslashes in code and output intact.
        normalized_name = name.replace("\\", "/")
        body = body.replace(name, normalized_name)
        resource = _safe_output_path(out.parent, normalized_name)
        resource.parent.mkdir(parents=True, exist_ok=True)
        resource.write_bytes(data)

    body = re.sub(r"(?m)^[ \t]+$", "", body)
    out.write_text(body, encoding="utf-8", newline="\n")


def build_one(src: Path) -> Path:
    """Build one executable notebook or copy one README navigation source."""
    rel = src.relative_to(SRC_DIR)
    out = OUT_DIR / rel
    out.parent.mkdir(parents=True, exist_ok=True)

    if src.name.casefold() == "readme.md":
        shutil.copyfile(src, out)
    else:
        _write_rendered_notebook(src, out)
    return out


def _selection(value: str | None) -> Path:
    selected = (SRC_DIR / value).resolve() if value else SRC_DIR.resolve()
    source_root = SRC_DIR.resolve()
    if selected != source_root and source_root not in selected.parents:
        raise ValueError(f"Selection must stay under {SRC_DIR}: {value}")
    return selected


def discover_sources(selection: str | None = None) -> list[Path]:
    """Return Markdown sources in deterministic order, including READMEs."""
    selected = _selection(selection)
    if selected.is_dir():
        return sorted(selected.rglob("*.md"))
    if selected.is_file() and selected.suffix.casefold() == ".md":
        return [selected]
    return []


def _local_target(document: Path, raw_target: str) -> Path | None:
    target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None
    path = unquote(parsed.path)
    if path.startswith("/examples/"):
        return ROOT / path.lstrip("/")
    if path.startswith("/"):
        return None
    return document.parent / Path(*PurePosixPath(path).parts)


def validate_links(documents: list[Path]) -> list[str]:
    """Report broken local Markdown links in generated documents."""
    broken: list[str] = []
    for document in documents:
        text = document.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK.finditer(text):
            target = _local_target(document, match.group(1))
            if target is not None and not target.exists():
                broken.append(
                    f"{document.relative_to(ROOT).as_posix()} -> {match.group(1)}"
                )
    return broken


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "selection",
        nargs="?",
        help="file or directory relative to notebooks/ (default: all)",
    )
    args = parser.parse_args(argv)

    try:
        sources = discover_sources(args.selection)
    except ValueError as exc:
        parser.error(str(exc))

    if not sources:
        print(f"No Markdown notebook sources found for {args.selection or 'notebooks/'}")
        return 0

    built: list[Path] = []
    failed: list[tuple[Path, Exception]] = []
    for src in sources:
        try:
            out = build_one(src)
            built.append(out)
            action = "COPY" if src.name.casefold() == "readme.md" else "OK  "
            print(f"{action} {src.relative_to(ROOT)} -> {out.relative_to(ROOT)}")
        except Exception as exc:  # noqa: BLE001 - continue and report every failure
            failed.append((src, exc))
            print(f"FAIL {src.relative_to(ROOT)}: {type(exc).__name__}: {exc}")

    broken = validate_links(built)
    for item in broken:
        print(f"LINK {item}")

    if failed or broken:
        print(
            f"\nConversion finished with {len(failed)} failed notebook(s) "
            f"and {len(broken)} broken local link(s)."
        )
        return 1

    print(f"\n{len(built)} Markdown file(s) generated successfully; local links are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
