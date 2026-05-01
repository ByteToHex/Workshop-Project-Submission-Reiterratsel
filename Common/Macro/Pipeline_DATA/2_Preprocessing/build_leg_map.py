"""
build_leg_map.py
----------------
Pre-step: enriches 2VALIDATE_CTF_bracket_market_tokens_with_parent_dates.csv
with two new columns:

  signed_leg_move_bps  – signed basis-point move for the leg, parsed from
                         group_item_title (e.g. "25 bps decrease" → -25)

  fomc_decision_date   – canonical FOMC decision day (second day of meeting),
                         cross-referenced from fomc_schedule.csv via the
                         year+month of parent_end_date_intended

  date_source          – provenance of fomc_decision_date:
                           "fomc_schedule"    – matched in fomc_schedule.csv
                           "intended_end_date" – fell back to parent_end_date_intended

  bps_parse_note       – blank unless the parser needed to flag something:
                           "cumulative_binary" – early "Above X.XX%" markets
                                                 (not mutually-exclusive legs)
                           "other_leg"         – "Other" catch-all legs
                           "unparsed"          – regex matched nothing; review
"""

import re
import calendar
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────

FOMC_SCHEDULE_CSV = r"d:\WS_NUS\REF_DATA\prediction-market-analysis\scripts\export\REF\fomc_schedule.csv"

VALIDATE_CSV      = r"d:\WS_NUS\REF_DATA\prediction-market-analysis\scripts\export\REF\2VALIDATE_CTF_bracket_market_tokens_with_parent_dates.csv"

OUTPUT_CSV        = r"d:\WS_NUS\REF_DATA\prediction-market-analysis\scripts\export\REF\2VALIDATE_CTF_REF_Bracket_ParentDate_LegMove_FomcDate.csv"


# ── Step 0 – Build FOMC decision-date lookup ─────────────────────────────────
# Decision day = second (last) day of the two-day meeting.
# Cross-month meetings (e.g. "April/May 30-1") land on day 1 of the second
# month; the lookup key is that decision year+month.

def _build_month_map() -> dict[str, int]:
    """Full name and 3-letter abbreviation → month number."""
    m: dict[str, int] = {}
    for num, name in enumerate(calendar.month_name):
        if not name:
            continue
        m[name.lower()]       = num
        m[name[:3].lower()]   = num
    return m

MONTH_MAP = _build_month_map()


def parse_fomc_schedule(path: str) -> dict[tuple[int, int], pd.Timestamp]:
    """
    Returns {(year, month_num): decision_timestamp} for every FOMC meeting
    in the schedule CSV.

    Handles:
    - Forward-filled Year column
    - Cross-month entries: "April/May", "Jan/Feb", "Oct/Nov"
    - Cross-month date spans: "30-1" (decision day = 1 of second month)
    - Single-day entries: "22" (notation votes)
    - Asterisks marking press-conference meetings (stripped)
    - Parenthetical notes in Month like "August (notation vote)" (stripped)
    """
    df = pd.read_csv(path, dtype=str)
    df["Year"] = df["Year"].ffill()

    schedule: dict[tuple[int, int], pd.Timestamp] = {}

    for _, row in df.iterrows():
        year_raw  = str(row["Year"]).strip()
        month_raw = str(row["Month"]).strip()
        date_raw  = str(row["Date"]).replace("*", "").strip()

        try:
            base_year = int(float(year_raw))
        except ValueError:
            continue

        # Strip parenthetical notes from month  e.g. "August (notation vote)"
        month_clean = re.sub(r"\s*\(.*?\)", "", month_raw).strip()

        # Slash months: "April/May" → first=April, second=May
        if "/" in month_clean:
            parts_m   = [p.strip().lower() for p in month_clean.split("/")]
            first_mo  = MONTH_MAP.get(parts_m[0])
            second_mo = MONTH_MAP.get(parts_m[1])
        else:
            first_mo  = MONTH_MAP.get(month_clean.lower())
            second_mo = first_mo

        if first_mo is None or second_mo is None:
            print(f"  [WARN] Could not parse month: '{month_raw}'")
            continue

        # Parse day range
        parts_d = date_raw.split("-")
        if len(parts_d) == 2:
            try:
                day1 = int(parts_d[0])
                day2 = int(parts_d[1])
            except ValueError:
                print(f"  [WARN] Could not parse date: '{date_raw}'")
                continue

            # Cross-month rollover: day2 < day1 (e.g. "30-1") AND single month listed
            # → decision day is day2 of second_mo; if "/" wasn't in month, advance month
            if day2 < day1 and "/" not in month_clean:
                decision_month = second_mo + 1 if second_mo < 12 else 1
                decision_year  = base_year + 1 if second_mo == 12 else base_year
            else:
                decision_month = second_mo
                decision_year  = base_year

            decision_day = day2

        else:
            # Single day (notation votes)
            try:
                decision_day = int(parts_d[0])
            except ValueError:
                print(f"  [WARN] Could not parse single day: '{date_raw}'")
                continue
            decision_month = first_mo
            decision_year  = base_year

        try:
            ts = pd.Timestamp(decision_year, decision_month, decision_day)
            key = (decision_year, decision_month)
            schedule[key] = ts
        except Exception as exc:
            print(f"  [WARN] Could not create timestamp for {decision_year}-{decision_month}-{decision_day}: {exc}")

    return schedule


# ── Step 3 – Parse signed_leg_move_bps from group_item_title ─────────────────

def parse_signed_leg_move_bps(title: str) -> tuple[int | None, str]:
    """
    Returns (signed_bps, note) where note is '' unless flagging is needed.

    Patterns handled:
      "Above X.XX% ... (N bps or more)"   → +N, note="cumulative_binary"
      "N+ bps decrease" / "N bps decrease" → -N
      "N+ bps increase" / "N bps increase" → +N
      "0 bps increase"                     → 0
      "No increase" / "No change"          → 0
      "Other"                              → None, note="other_leg"
      anything else                        → None, note="unparsed"
    """
    t = title.strip().lower()

    # "Other" catch-all leg
    if re.fullmatch(r"other", t):
        return None, "other_leg"

    # "No increase" / "No change" / "No Change"
    if re.search(r"\bno\s+(increase|change)\b", t):
        return 0, ""

    # "0 bps increase" (explicit zero)
    if re.search(r"\b0\s*bps\s+increase\b", t):
        return 0, ""

    # Cumulative binary: "Above X.XX% ... (N bps or more)"
    m = re.search(r"\((\d+)\s*bps\s+or\s+more\)", t)
    if m:
        return int(m.group(1)), "cumulative_binary"

    # Extract magnitude: "N+" or "N" followed by "bps"
    m = re.search(r"(\d+)\+?\s*bps", t)
    if not m:
        return None, "unparsed"

    n = int(m.group(1))

    if re.search(r"\bdecrease\b|\bcut\b|\breduction\b", t):
        return -n, ""
    if re.search(r"\bincrease\b|\bhike\b|\braise\b", t):
        return +n, ""

    # Magnitude found but direction ambiguous
    return None, "unparsed"


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # 0. Build FOMC schedule lookup
    print("Parsing FOMC schedule …")
    schedule = parse_fomc_schedule(FOMC_SCHEDULE_CSV)
    print(f"  {len(schedule)} meeting dates loaded:")
    for k, v in sorted(schedule.items()):
        print(f"    {k} -> {v.date()}")

    # 1. Load 2VALIDATE CSV
    print(f"\nLoading validate CSV …")
    df = pd.read_csv(VALIDATE_CSV, dtype=str)
    print(f"  {len(df)} rows loaded, {df['parent_event_id'].nunique()} unique events")

    # 2. Cross-reference fomc_decision_date via parent_end_date_intended
    def lookup_fomc_date(intended_str: str) -> tuple[str, str]:
        """Returns (fomc_decision_date_str, date_source)."""
        try:
            ts = pd.Timestamp(intended_str)
        except Exception:
            return "", "intended_end_date"

        key = (ts.year, ts.month)
        if key in schedule:
            return schedule[key].strftime("%Y-%m-%d"), "fomc_schedule"

        # Fallback: use intended date itself (strip time component)
        return ts.strftime("%Y-%m-%d"), "intended_end_date"

    fomc_results = df["parent_end_date_intended"].apply(
        lambda v: pd.Series(lookup_fomc_date(v), index=["fomc_decision_date", "date_source"])
    )
    df = pd.concat([df, fomc_results], axis=1)

    # 3. Parse signed_leg_move_bps
    bps_results = df["group_item_title"].apply(
        lambda t: pd.Series(parse_signed_leg_move_bps(str(t)), index=["signed_leg_move_bps", "bps_parse_note"])
    )
    df = pd.concat([df, bps_results], axis=1)

    # Convert to nullable Int64 so NaN is preserved cleanly (not as float)
    df["signed_leg_move_bps"] = pd.to_numeric(df["signed_leg_move_bps"], errors="coerce")
    df["signed_leg_move_bps"] = df["signed_leg_move_bps"].astype("Int64")

    # 4. QA summary
    print("\nQA summary:")
    print(f"  fomc_schedule matches : {(df['date_source'] == 'fomc_schedule').sum()}")
    print(f"  intended_end_date fb  : {(df['date_source'] == 'intended_end_date').sum()}")
    print(f"  bps parsed ok         : {df['signed_leg_move_bps'].notna().sum()}")
    print(f"  cumulative_binary     : {(df['bps_parse_note'] == 'cumulative_binary').sum()}")
    print(f"  other_leg             : {(df['bps_parse_note'] == 'other_leg').sum()}")
    print(f"  unparsed              : {(df['bps_parse_note'] == 'unparsed').sum()}")

    unparsed = df[df["bps_parse_note"] == "unparsed"]["group_item_title"].unique()
    if len(unparsed):
        print(f"\n  [REVIEW] Unparsed titles:")
        for t in unparsed:
            print(f"    {t!r}")

    fallback_events = df[df["date_source"] == "intended_end_date"]["parent_event_slug"].unique()
    if len(fallback_events):
        print(f"\n  [REVIEW] Events using intended_end_date fallback:")
        for s in fallback_events:
            print(f"    {s}")

    # 5. Write output
    print(f"\nWriting -> {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
