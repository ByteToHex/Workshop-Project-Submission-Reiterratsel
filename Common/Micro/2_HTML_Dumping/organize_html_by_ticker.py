"""
Move TradingView HTML dumps into subfolders named after the ticker in the leading
``[TICKER]_...`` filename prefix (see DumpHtml / dumper scripts).

Example: ``[A17U]_2-cash_flow(N_A)_Annual.html`` → ``<src>/A17U/``.

By default only files directly under ``src`` are moved; use ``--recursive`` to collect
matches from nested directories (destinations are always ``<src>/<ticker>/``).
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

# First bracket group at the start of the basename (ticker from TradingView path)
TICKER_PREFIX = re.compile(r"^\[([^\]]*)\]")


def _sanitize_dir_name(ticker: str) -> str:
    """Make a single path segment safe on Windows and Unix."""
    s = ticker.strip()
    for ch in '<>:"/\\|?*':
        s = s.replace(ch, "_")
    s = s.rstrip(" .")
    return s if s else "_empty"


def _unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem, suffix = dest.stem, dest.suffix
    parent = dest.parent
    for n in range(2, 10_000):
        candidate = parent / f"{stem}__dup{n}{suffix}"
        if not candidate.exists():
            return candidate
    raise OSError(f"Could not find unique name near {dest}")


def _iter_files(src: Path, recursive: bool, exts: frozenset[str]) -> list[Path]:
    if recursive:
        out: list[Path] = []
        for p in src.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in exts:
                continue
            out.append(p)
        return sorted(out)

    return sorted(
        p
        for p in src.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )


def _parse_ticker(name: str) -> str | None:
    m = TICKER_PREFIX.match(name)
    if not m:
        return None
    raw = m.group(1).strip()
    return raw if raw else None


def run(
    src: Path,
    *,
    dry_run: bool,
    recursive: bool,
    exts: frozenset[str],
) -> int:
    src = src.resolve()
    if not src.is_dir():
        print(f"Error: not a directory: {src}")
        return 1

    files = _iter_files(src, recursive, exts)
    moved = 0
    skipped_no_ticker = 0
    skipped_same = 0

    for path in files:
        ticker_raw = _parse_ticker(path.name)
        if ticker_raw is None:
            skipped_no_ticker += 1
            continue

        sub = _sanitize_dir_name(ticker_raw)
        dest_dir = (src / sub).resolve()
        if path.parent.resolve() == dest_dir:
            skipped_same += 1
            continue

        dest = _unique_dest(dest_dir / path.name)
        if dry_run:
            print(f"DRY-RUN: {path} -> {dest}")
        else:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(dest))
            print(f"Moved: {path.name} -> {dest}")
        moved += 1

    print(
        f"Done. moved={moved}, skipped_no_ticker_pattern={skipped_no_ticker}, "
        f"skipped_already_placed={skipped_same}, files_scanned={len(files)}"
    )
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Group HTML dumps into subfolders by leading [TICKER] in the filename."
    )
    ap.add_argument(
        "src",
        nargs="?",
        type=Path,
        default=Path.cwd(),
        help="Directory containing dumped HTML (default: current directory)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned moves without moving files",
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="Include HTML files in subdirectories (flat output under src/<ticker>/)",
    )
    ap.add_argument(
        "--ext",
        default=".html,.htm",
        help="Comma-separated file extensions to include (default: .html,.htm)",
    )
    args = ap.parse_args()
    raw_exts = [x.strip().lower() for x in args.ext.split(",") if x.strip()]
    exts = frozenset(x if x.startswith(".") else f".{x}" for x in raw_exts)
    raise SystemExit(
        run(args.src, dry_run=args.dry_run, recursive=args.recursive, exts=exts)
    )


if __name__ == "__main__":
    main()
