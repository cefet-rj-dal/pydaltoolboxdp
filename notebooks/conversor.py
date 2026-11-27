"""Execute Markdown notebook sources and render them under ``examples/``."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path, PurePosixPath

import jupytext
from nbconvert import MarkdownExporter
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "notebooks"
OUTPUT = ROOT / "examples"


def sources(selection: str | None) -> list[Path]:
    selected = (SOURCE / selection).resolve() if selection else SOURCE.resolve()
    root = SOURCE.resolve()
    if selected != root and root not in selected.parents:
        raise ValueError("selection must stay inside notebooks/")
    if selected.is_file() and selected.suffix.lower() == ".md":
        return [selected]
    return sorted(selected.rglob("*.md")) if selected.is_dir() else []


def safe_resource(document_dir: Path, name: str) -> Path:
    normalized = name.replace("\\", "/")
    target = (document_dir / Path(*PurePosixPath(normalized).parts)).resolve()
    output = OUTPUT.resolve()
    if target != output and output not in target.parents:
        raise ValueError(f"unsafe generated resource: {name}")
    return target


def build(source: Path) -> Path:
    destination = OUTPUT / source.relative_to(SOURCE)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.name.lower() == "readme.md":
        shutil.copyfile(source, destination)
        return destination

    notebook = jupytext.read(source)
    ExecutePreprocessor(timeout=300, kernel_name="python3").preprocess(
        notebook, {"metadata": {"path": str(source.parent)}}
    )
    figure_dir = PurePosixPath("fig", source.stem).as_posix()
    body, resources = MarkdownExporter().from_notebook_node(
        notebook, resources={"output_files_dir": figure_dir}
    )
    for name, data in resources.get("outputs", {}).items():
        normalized = name.replace("\\", "/")
        body = body.replace(name, normalized)
        resource = safe_resource(destination.parent, normalized)
        resource.parent.mkdir(parents=True, exist_ok=True)
        resource.write_bytes(data)
    destination.write_text(body, encoding="utf-8", newline="\n")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", nargs="?")
    args = parser.parse_args()
    selected = sources(args.selection)
    if not selected:
        print("No Markdown notebook sources found")
        return 0
    for source in selected:
        destination = build(source)
        print(f"OK {source.relative_to(ROOT)} -> {destination.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
