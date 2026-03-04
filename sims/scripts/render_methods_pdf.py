#!/usr/bin/env python3
r"""Render sims/docs Markdown files into TeX-typeset PDFs under sims/docs/docs_pdf.

Uses Pandoc + LaTeX (`pdflatex`) so equations and formatting are rendered as
publication-style document output rather than plain wrapped text.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render sims docs markdown files to PDF.")
    parser.add_argument("--docs-dir", default="sims/docs", help="Directory containing markdown docs.")
    parser.add_argument("--out-dir", default="sims/docs/docs_pdf", help="Output directory for generated PDFs.")
    return parser.parse_args()


def _markdown_files(docs_dir: Path) -> list[Path]:
    files = [
        path
        for path in docs_dir.rglob("*.md")
        if "docs_pdf" not in path.parts and path.is_file()
    ]
    return sorted(files)


def _require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required executable not found on PATH: {name}")


def render_markdown_to_pdf(md_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "pandoc",
        str(md_path),
        "--from",
        "markdown+tex_math_dollars",
        "--to",
        "pdf",
        "--pdf-engine",
        "pdflatex",
        "--variable",
        "geometry:margin=1in",
        "--output",
        str(out_path),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    out_dir = Path(args.out_dir)

    _require_executable("pandoc")
    _require_executable("pdflatex")

    if not docs_dir.exists():
        raise FileNotFoundError(f"docs dir not found: {docs_dir}")

    md_files = _markdown_files(docs_dir)
    if not md_files:
        raise RuntimeError(f"No markdown files found under: {docs_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for md_path in md_files:
        out_name = md_path.stem + ".pdf"
        out_path = out_dir / out_name
        render_markdown_to_pdf(md_path=md_path, out_path=out_path)
        print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
