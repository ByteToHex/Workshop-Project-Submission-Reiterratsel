from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
ENV_PYTHON = Path(r"C:\ProgramData\anaconda3\envs\env\python.exe")
ENV_STREAMLIT = Path(r"C:\ProgramData\anaconda3\envs\env\Scripts\streamlit.exe")
BUILD_SCRIPT = ROOT_DIR / "Common" / "Micro" / "5_Model_KG" / "build_reitteratsel_pipeline.py"
APP_SCRIPT = ROOT_DIR / "Common" / "Frontend" / "reitteratsel_app.py"


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def ensure_project_runtime() -> None:
    actual_python = Path(sys.executable).resolve()
    expected_python = ENV_PYTHON.resolve()
    if actual_python != expected_python:
        raise RuntimeError(
            "This orchestrator must be run with the project-standard Anaconda env Python.\n"
            f"Expected: {expected_python}\n"
            f"Actual:   {actual_python}"
        )


def run_build() -> None:
    subprocess.run(
        [str(ENV_PYTHON), str(BUILD_SCRIPT)],
        cwd=str(ROOT_DIR),
        check=True,
    )


def run_app() -> None:
    subprocess.run(
        [str(ENV_STREAMLIT), "run", str(APP_SCRIPT)],
        cwd=str(ROOT_DIR),
        check=True,
    )


def main() -> None:
    ensure_project_runtime()
    require_path(ENV_PYTHON, "Project Python interpreter")
    require_path(ENV_STREAMLIT, "Project Streamlit executable")
    require_path(BUILD_SCRIPT, "REITterratsel build script")
    require_path(APP_SCRIPT, "REITterratsel app script")

    print("Running REITterratsel build pipeline...")
    run_build()
    print("Launching REITterratsel Streamlit app...")
    run_app()


if __name__ == "__main__":
    main()
