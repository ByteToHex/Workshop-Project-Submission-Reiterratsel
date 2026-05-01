"""
SOURCE: "...\scripts\export\REF_Preprocessing\Step4_XGBoost\sample_rates\parse_mas_sora.py"
parse_mas_sora.py

Parses the two MAS Domestic Interest Rates CSVs stored in the shared
Consolidated IO/SRC tree:
  - CSV_MAS/Input/DomesticInterestRates_Idx14_SORA.csv
  - CSV_MAS/Input/DomesticInterestRates_idx17_SORA3MthCompounded.csv

Outputs:
  - CSV_MAS/Output/sora_daily.csv
  - CSV_MAS/Output/sora_3m_daily.csv

This keeps the extracted Consolidated folder self-contained:
the parser script lives with the other stage-4/5 prep scripts, while the
raw MAS files live in the shared IO/SRC area.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


SCRIPT_DIR = Path(__file__).resolve().parent
CONSOLIDATED_ROOT = SCRIPT_DIR.parents[1]
IO_SRC_DIR = CONSOLIDATED_ROOT / "IO" / "SRC"
RAW_DIR = IO_SRC_DIR / "CSV_MAS" / "Input"
RATES_DIR = IO_SRC_DIR / "CSV_MAS" / "Output"

SORA_RAW = RAW_DIR / "DomesticInterestRates_Idx14_SORA.csv"
SORA_3M_RAW = RAW_DIR / "DomesticInterestRates_idx17_SORA3MthCompounded.csv"

OUT_SORA = RATES_DIR / "sora_daily.csv"
OUT_SORA_3M = RATES_DIR / "sora_3m_daily.csv"

MONTH_MAP = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def parse_mas_file(path: Path, value_col_name: str) -> pd.DataFrame:
    rows: list[tuple[int, int, int, float]] = []
    current_year: int | None = None
    current_month: int | None = None

    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            if lineno <= 7:
                continue

            line = raw.rstrip("\n")

            if line.startswith('"'):
                break
            if not line.strip():
                continue
            if line.startswith("SORA Value Date"):
                continue

            parts = line.split(",")
            if len(parts) < 5:
                continue

            year_str, month_str, day_str, _pub_date, value_str = (
                parts[0].strip(),
                parts[1].strip(),
                parts[2].strip(),
                parts[3].strip(),
                parts[4].strip(),
            )

            if year_str:
                current_year = int(year_str)
            if month_str:
                current_month = MONTH_MAP[month_str]

            if current_year is None or current_month is None or not day_str:
                continue

            try:
                day = int(day_str)
                value = float(value_str)
            except ValueError:
                continue

            rows.append((current_year, current_month, day, value))

    df = pd.DataFrame(rows, columns=["year", "month", "day", value_col_name])
    df["value_date"] = pd.to_datetime(df[["year", "month", "day"]])
    df = df[["value_date", value_col_name]].copy()
    df = df.sort_values("value_date").drop_duplicates("value_date").reset_index(drop=True)
    return df


def require_inputs() -> None:
    missing = [str(path) for path in [SORA_RAW, SORA_3M_RAW] if not path.exists()]
    if missing:
        print("[ERROR] Missing MAS source CSV(s):")
        for path in missing:
            print(f"  - {path}")
        sys.exit(1)


def main() -> None:
    require_inputs()
    RATES_DIR.mkdir(parents=True, exist_ok=True)

    print("Parsing overnight SORA ...")
    sora = parse_mas_file(SORA_RAW, "sora_level")
    print(
        f"  {len(sora):,} business-day rows | "
        f"{sora['value_date'].min().date()} to {sora['value_date'].max().date()}"
    )

    print("Parsing 3-month compounded SORA ...")
    sora_3m = parse_mas_file(SORA_3M_RAW, "sora_3m")
    print(
        f"  {len(sora_3m):,} business-day rows | "
        f"{sora_3m['value_date'].min().date()} to {sora_3m['value_date'].max().date()}"
    )

    sora.to_csv(OUT_SORA, index=False)
    sora_3m.to_csv(OUT_SORA_3M, index=False)
    print(f"Saved: {OUT_SORA}")
    print(f"Saved: {OUT_SORA_3M}")


if __name__ == "__main__":
    main()
