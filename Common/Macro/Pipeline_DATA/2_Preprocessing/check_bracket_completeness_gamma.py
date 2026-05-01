"""
Validate bracket completeness of exported Parquet CSVs against Gamma event ground truth.

Expected workflow:
  1) Export with Step_01 script (events or markets mode).
  2) Run this script on the exported CSV.
  3) Inspect audit CSVs + summary JSON under scripts/util/out_Markets/.
     When `out_Trades/SCHEMA/3PROBE_market_token_map.csv` exists, also writes
     `2VALIDATE_bracket_market_tokens.csv` (per expected market_slug + token_id rows,
     ordered by `1EXTRACT_fed_parquet_events.csv` min_end_date where possible,
     with Gamma group_item_title / group_item_threshold).

This script supports either:
  - market-level CSVs (contains a `slug` column), or
  - event-level CSVs (contains `outcome_market_slugs` with ';' separated slugs).

---
It’s okay to run from the button **if** you want default behavior:
- input: `scripts/util/out_Markets/1EXTRACT_fed_parquet_events.csv`
- output files written/overwritten in `scripts/util/out_Markets/`
- makes live HTTP calls to Gamma API

If you need a different input CSV or custom flags, run from terminal with args (or set a launch config), e.g.:

`python scripts/util/Step_02_CoverageCheck/check_bracket_completeness_gamma.py --input-csv your_file.csv`
---
Exit behavior (for CI and automation):
- Returns 0 when coverage checks complete and no events are incomplete.
- Returns 7 when one or more events are incomplete.
- Pass `--allow-incomplete` to keep legacy behavior (still write all outputs, but return 0).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.util.trade_probe_exports import export_market_token_map_and_trade_coverage

_UTIL_ROOT = Path(__file__).resolve().parents[1]
_OUT_DIR = _UTIL_ROOT / "out_Markets"
_STEP01_PREFIX = "1EXTRACT"
_STEP02_PREFIX = "2VALIDATE"
_DEFAULT_INPUT = _OUT_DIR / f"{_STEP01_PREFIX}_fed_parquet_events.csv"
_GAMMA_BASE = "https://gamma-api.parquet.com"


def _csv_path(p: Path) -> Path:
    return p if p.is_absolute() else (_OUT_DIR / p)


def _fetch_json(
    url: str,
    *,
    timeout_s: float,
    max_retries: int,
    retry_sleep_s: float,
) -> Any:
    last_err: Exception | None = None
    req = Request(url, headers={"Accept": "application/json", "User-Agent": "bracket-check/1.0"})
    for attempt in range(max_retries + 1):
        try:
            with urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            if attempt >= max_retries:
                break
            time.sleep(retry_sleep_s * (attempt + 1))
    raise RuntimeError(f"GET failed: {url} ({last_err})")


def _read_input_market_slugs(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return [], []

    slugs: list[str] = []
    for r in rows:
        if "slug" in r and (r.get("slug") or "").strip():
            slugs.append((r.get("slug") or "").strip())
            continue
        raw = (r.get("outcome_market_slugs") or "").strip()
        if raw:
            for s in raw.split(";"):
                s2 = s.strip()
                if s2:
                    slugs.append(s2)
    # stable-unique
    seen: set[str] = set()
    uniq = []
    for s in slugs:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return rows, uniq


def _write_csv(path: Path, rows: list[dict[str, Any]], header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def _first_nonempty_str(d: dict[str, Any], keys: list[str]) -> str:
    for k in keys:
        v = d.get(k)
        s = str(v or "").strip()
        if s:
            return s
    return ""


def _parse_dt_iso(value: str) -> datetime | None:
    v = (value or "").strip()
    if not v:
        return None
    try:
        return datetime.fromisoformat(v)
    except ValueError:
        return None


def _read_extract_min_end_date_maps(path: Path) -> tuple[dict[str, datetime], dict[str, datetime]]:
    """
    From Step 01 `1EXTRACT_*.csv`, accumulate earliest min_end_date per parent_event_slug
    and per outcome market slug (same date column `sort_csv_by_end_date` prefers).
    """
    parent_min: dict[str, datetime] = {}
    slug_min: dict[str, datetime] = {}
    if not path.is_file():
        return parent_min, slug_min
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            raw = (row.get("min_end_date") or "").strip()
            if not raw:
                continue
            dt = _parse_dt_iso(raw)
            if dt is None:
                continue
            ps = (row.get("parent_event_slug") or "").strip()
            if ps:
                if ps not in parent_min or dt < parent_min[ps]:
                    parent_min[ps] = dt
            for s in (row.get("outcome_market_slugs") or "").split(";"):
                s2 = s.strip()
                if not s2:
                    continue
                if s2 not in slug_min or dt < slug_min[s2]:
                    slug_min[s2] = dt
    return parent_min, slug_min


def _bracket_parent_sort_dt(
    pkey: str,
    expected_slugs: set[str],
    parent_min: dict[str, datetime],
    slug_min: dict[str, datetime],
    gamma_event_end_iso: str,
) -> datetime | None:
    """Order key aligned with 1EXTRACT: min of relevant min_end_date rows, else Gamma event endDate."""
    cands: list[datetime] = []
    if pkey in parent_min:
        cands.append(parent_min[pkey])
    for s in expected_slugs:
        if s in slug_min:
            cands.append(slug_min[s])
    if cands:
        return min(cands)
    return _parse_dt_iso(gamma_event_end_iso)


def _read_market_token_map(path: Path) -> dict[str, list[str]]:
    """slug -> ordered list of token_id (multiple rows per slug = multiple outcomes)."""
    if not path.is_file():
        return {}
    out: dict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            slug = str(row.get("market_slug") or "").strip()
            tid = str(row.get("token_id") or "").strip()
            if not slug or not tid:
                continue
            out[slug].append(tid)
    return dict(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-csv",
        type=Path,
        default=_DEFAULT_INPUT,
        help="Input CSV path (absolute or relative to scripts/util/out_Markets).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_OUT_DIR,
        help="Audit output directory (absolute or relative to scripts/util/out_Markets).",
    )
    p.add_argument("--timeout-s", type=float, default=20.0, help="HTTP timeout in seconds.")
    p.add_argument("--retries", type=int, default=2, help="Retries per request.")
    p.add_argument("--retry-sleep-s", type=float, default=0.75, help="Base retry sleep.")
    p.add_argument("--sleep-ms", type=int, default=60, help="Sleep between API calls.")
    p.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print an inline progress counter every N sleep intervals (0 disables).",
    )
    p.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Exit 0 even when event completeness is incomplete.",
    )
    p.add_argument(
        "--token-map-csv",
        type=Path,
        default=_UTIL_ROOT / "out_Trades" / "SCHEMA" / "3PROBE_market_token_map.csv",
        help=(
            "Optional CSV with columns market_slug, token_id (e.g. probe output). "
            "If the file exists, writes a per-bracket market_slug/token_id detail CSV."
        ),
    )
    p.add_argument(
        "--no-token-map",
        action="store_true",
        help="Do not read --token-map-csv or write the bracket token detail CSV.",
    )
    p.add_argument(
        "--skip-trade-probe-export",
        action="store_true",
        help="Skip preliminary Step 03-style exports for market token map and trade coverage.",
    )
    p.add_argument(
        "--trade-probe-data-dir",
        type=Path,
        default=Path("data/parquet"),
        help="Data root containing markets/ and trades/ parquet for preliminary trade probe export.",
    )
    p.add_argument(
        "--trade-probe-output-prefix",
        type=str,
        default="3PROBE",
        help="Output prefix for preliminary trade probe files under scripts/util/out_Trades/SCHEMA.",
    )
    p.add_argument(
        "--trade-probe-trade-sample-files",
        type=int,
        default=None,
        help="Optional limit for trades_*.parquet files in preliminary trade probe export.",
    )
    p.add_argument(
        "--trade-probe-market-sample-files",
        type=int,
        default=None,
        help="Optional limit for markets_*.parquet files in preliminary trade probe export.",
    )
    p.add_argument(
        "--trade-probe-heartbeat-sec",
        type=float,
        default=5.0,
        help="Heartbeat cadence in seconds for preliminary trade probe stages.",
    )
    p.add_argument(
        "--extract-sort-csv",
        type=Path,
        default=Path(f"{_STEP01_PREFIX}_fed_parquet_events.csv"),
        help=(
            "Step 01 1EXTRACT CSV (under out_Markets unless absolute). "
            "Used to order parent ledgers by min_end_date like csv_sorting / parquet_fed_events_export."
        ),
    )
    args = p.parse_args(argv)

    input_csv = _csv_path(args.input_csv)
    out_dir = _csv_path(args.out_dir)
    extract_sort_csv = _csv_path(args.extract_sort_csv)
    token_map_path = args.token_map_csv if not args.no_token_map else None
    default_token_map = _UTIL_ROOT / "out_Trades" / "SCHEMA" / "3PROBE_market_token_map.csv"
    if (
        token_map_path is not None
        and token_map_path == default_token_map
        and args.trade_probe_output_prefix != "3PROBE"
    ):
        token_map_path = _UTIL_ROOT / "out_Trades" / "SCHEMA" / (
            f"{args.trade_probe_output_prefix}_market_token_map.csv"
        )

    if not input_csv.is_file():
        print(f"Input CSV not found: {input_csv}")
        return 1

    if not args.skip_trade_probe_export:
        print("pre-step: generating trade probe exports (token map + trade coverage)")
        try:
            probe_summary = export_market_token_map_and_trade_coverage(
                input_csv=input_csv,
                data_dir=args.trade_probe_data_dir,
                output_prefix=args.trade_probe_output_prefix,
                trade_sample_files=args.trade_probe_trade_sample_files,
                market_sample_files=args.trade_probe_market_sample_files,
                heartbeat_sec=args.trade_probe_heartbeat_sec,
            )
            print(f"pre-step wrote: {probe_summary['output_market_token_map_csv']}")
            print(f"pre-step wrote: {probe_summary['output_market_trade_coverage_csv']}")
        except Exception as e:
            print(f"pre-step failed: {e}")
            return 8

    input_rows, observed_slugs = _read_input_market_slugs(input_csv)
    if not observed_slugs:
        print("No market slugs found in input CSV (`slug` or `outcome_market_slugs`).")
        return 2

    print(f"input_csv:        {input_csv}")
    print(f"extract_sort_csv:   {extract_sort_csv}")
    print(f"input_rows:         {len(input_rows)}")
    print(f"observed_slugs:     {len(observed_slugs)}")

    unresolved_rows: list[dict[str, Any]] = []
    market_mismatch_rows: list[dict[str, Any]] = []
    printed_progress_dot = False
    sleep_calls = 0

    # slug -> market payload fields (normalized)
    market_info: dict[str, dict[str, Any]] = {}
    # parent_key -> meta
    parent_meta: dict[str, dict[str, Any]] = {}
    # parent_key -> observed slugs
    observed_by_parent: dict[str, set[str]] = defaultdict(set)

    def _sleep_if_needed() -> None:
        nonlocal printed_progress_dot, sleep_calls
        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)
            # Heartbeat during long API loops so users can see forward progress.
            print(".", end="", flush=True)
            printed_progress_dot = True
            sleep_calls += 1
            if args.progress_every > 0 and sleep_calls % args.progress_every == 0:
                print(f" {sleep_calls}", end="", flush=True)

    for slug in observed_slugs:
        url = f"{_GAMMA_BASE}/markets/slug/{slug}"
        try:
            m = _fetch_json(
                url,
                timeout_s=args.timeout_s,
                max_retries=args.retries,
                retry_sleep_s=args.retry_sleep_s,
            )
        except Exception as e:
            unresolved_rows.append(
                {
                    "slug": slug,
                    "stage": "market_lookup",
                    "reason": str(e),
                    "parent_event_id": "",
                    "parent_event_slug": "",
                }
            )
            _sleep_if_needed()
            continue

        events = m.get("events") or []
        ev0 = events[0] if events else {}
        ev_id = str(ev0.get("id") or "").strip()
        ev_slug = str(ev0.get("slug") or "").strip()
        parent_key = ev_slug or f"id:{ev_id}" if ev_id else ""

        market_info[slug] = {
            "market_slug": str(m.get("slug") or slug),
            "market_id": str(m.get("id") or ""),
            "condition_id": str(m.get("conditionId") or ""),
            "resolution_source": str(m.get("resolutionSource") or ""),
            "end_date": str(m.get("endDate") or ""),
            "group_item_title": str(m.get("groupItemTitle") or ""),
            "group_item_threshold": str(m.get("groupItemThreshold") or ""),
            "parent_event_id": ev_id,
            "parent_event_slug": ev_slug,
            "parent_event_title": str(ev0.get("title") or ""),
            "parent_series_slug": str(ev0.get("seriesSlug") or ""),
            "parent_key": parent_key,
        }

        if not parent_key:
            unresolved_rows.append(
                {
                    "slug": slug,
                    "stage": "parent_linkage",
                    "reason": "market found but events[0].id/slug missing",
                    "parent_event_id": ev_id,
                    "parent_event_slug": ev_slug,
                }
            )
        else:
            observed_by_parent[parent_key].add(slug)
            if parent_key not in parent_meta:
                parent_meta[parent_key] = {
                    "parent_event_id": ev_id,
                    "parent_event_slug": ev_slug,
                    "parent_event_title": str(ev0.get("title") or ""),
                    "parent_series_slug": str(ev0.get("seriesSlug") or ""),
                }
        _sleep_if_needed()

    # Load event-level ground truth
    expected_by_parent: dict[str, set[str]] = defaultdict(set)
    event_child_by_parent_slug: dict[str, dict[str, dict[str, str]]] = {}
    event_consistency_by_parent: dict[str, dict[str, Any]] = {}
    parent_market_rows: list[dict[str, Any]] = []

    for pkey, meta in parent_meta.items():
        ev_slug = (meta.get("parent_event_slug") or "").strip()
        ev_id = (meta.get("parent_event_id") or "").strip()
        if ev_slug:
            url = f"{_GAMMA_BASE}/events/slug/{ev_slug}"
        elif ev_id:
            url = f"{_GAMMA_BASE}/events?{urlencode({'id': ev_id})}"
        else:
            unresolved_rows.append(
                {
                    "slug": "",
                    "stage": "event_lookup",
                    "reason": f"missing event id/slug for parent key {pkey}",
                    "parent_event_id": ev_id,
                    "parent_event_slug": ev_slug,
                }
            )
            continue

        try:
            ev_data = _fetch_json(
                url,
                timeout_s=args.timeout_s,
                max_retries=args.retries,
                retry_sleep_s=args.retry_sleep_s,
            )
        except Exception as e:
            unresolved_rows.append(
                {
                    "slug": "",
                    "stage": "event_lookup",
                    "reason": str(e),
                    "parent_event_id": ev_id,
                    "parent_event_slug": ev_slug,
                }
            )
            _sleep_if_needed()
            continue

        ev = ev_data[0] if isinstance(ev_data, list) and ev_data else ev_data
        markets = ev.get("markets") or []
        child_map: dict[str, dict[str, str]] = {}

        end_dates: set[str] = set()
        res_sources: set[str] = set()
        thresholds: list[str] = []
        titles: list[str] = []
        for ch in markets:
            ch_slug = str(ch.get("slug") or "").strip()
            if not ch_slug:
                continue
            child_map[ch_slug] = {
                "market_id": str(ch.get("id") or ""),
                "condition_id": str(ch.get("conditionId") or ""),
                "end_date": str(ch.get("endDate") or ""),
                "resolution_source": str(ch.get("resolutionSource") or ""),
                "group_item_title": str(ch.get("groupItemTitle") or ""),
                "group_item_threshold": str(ch.get("groupItemThreshold") or ""),
                "closed": "" if ch.get("closed") is None else str(ch.get("closed")),
                "created_at": str(ch.get("createdAt") or ""),
                "_fetched_at": str(
                    ch.get("_fetchedAt")
                    or ch.get("_fetched_at")
                    or ch.get("fetchedAt")
                    or ""
                ),
            }
            expected_by_parent[pkey].add(ch_slug)
            if child_map[ch_slug]["end_date"]:
                end_dates.add(child_map[ch_slug]["end_date"])
            if child_map[ch_slug]["resolution_source"]:
                res_sources.add(child_map[ch_slug]["resolution_source"])
            if child_map[ch_slug]["group_item_threshold"]:
                thresholds.append(child_map[ch_slug]["group_item_threshold"])
            if child_map[ch_slug]["group_item_title"]:
                titles.append(child_map[ch_slug]["group_item_title"])

        event_child_by_parent_slug[pkey] = child_map
        event_consistency_by_parent[pkey] = {
            "unique_end_dates": len(end_dates),
            "unique_resolution_sources": len(res_sources),
            "dup_thresholds": len(thresholds) - len(set(thresholds)),
            "dup_titles": len(titles) - len(set(titles)),
        }
        parent_market_rows.append(
            {
                "parent_market_id": _first_nonempty_str(ev, ["id"]),
                "parent_market_slug": _first_nonempty_str(ev, ["slug"]),
                "meeting_title": _first_nonempty_str(ev, ["title"]),
                "meeting_series_slug": _first_nonempty_str(ev, ["seriesSlug"]),
                "meeting_key": pkey,
                "market_end_date_intended": _first_nonempty_str(ev, ["endDate"]),
                "market_end_date_actual_resolution": _first_nonempty_str(
                    ev,
                    [
                        "resolveDate",
                        "resolutionDate",
                        "resolvedAt",
                        "closedTime",
                        "closedAt",
                    ],
                ),
                "event_status": _first_nonempty_str(ev, ["status"]),
                "expected_n_markets": len(child_map),
                "observed_n_markets": len(observed_by_parent.get(pkey, set())),
                "missing_market_slugs": "",
                "extra_market_slugs": "",
            }
        )
        _sleep_if_needed()

    # Event completeness rows
    completeness_rows: list[dict[str, Any]] = []
    for pkey in sorted(set(parent_meta.keys()) | set(expected_by_parent.keys())):
        observed = observed_by_parent.get(pkey, set())
        expected = expected_by_parent.get(pkey, set())
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        matched = len(observed & expected)
        exp_n = len(expected)
        obs_n = len(observed)
        ratio = (matched / exp_n) if exp_n > 0 else 0.0
        meta = parent_meta.get(pkey, {})
        cons = event_consistency_by_parent.get(
            pkey,
            {
                "unique_end_dates": 0,
                "unique_resolution_sources": 0,
                "dup_thresholds": 0,
                "dup_titles": 0,
            },
        )
        completeness_rows.append(
            {
                "parent_key": pkey,
                "parent_event_id": meta.get("parent_event_id", ""),
                "parent_event_slug": meta.get("parent_event_slug", ""),
                "parent_event_title": meta.get("parent_event_title", ""),
                "parent_series_slug": meta.get("parent_series_slug", ""),
                "expected_n_markets": exp_n,
                "observed_n_markets": obs_n,
                "matched_n_markets": matched,
                "completeness_ratio": f"{ratio:.6f}",
                "missing_slugs": "; ".join(missing),
                "extra_slugs": "; ".join(extra),
                "unique_end_dates_in_event": cons["unique_end_dates"],
                "unique_resolution_sources_in_event": cons["unique_resolution_sources"],
                "duplicate_group_item_thresholds": cons["dup_thresholds"],
                "duplicate_group_item_titles": cons["dup_titles"],
            }
        )
    parent_market_row_by_key = {str(r.get("meeting_key") or ""): r for r in parent_market_rows}
    for r in completeness_rows:
        pkey = str(r.get("parent_key") or "")
        pmr = parent_market_row_by_key.get(pkey)
        if pmr is None:
            continue
        pmr["expected_n_markets"] = r.get("expected_n_markets", 0)
        pmr["observed_n_markets"] = r.get("observed_n_markets", 0)
        pmr["missing_market_slugs"] = r.get("missing_slugs", "")
        pmr["extra_market_slugs"] = r.get("extra_slugs", "")

    # Per-parent expected market slugs + token_ids (from optional probe map)
    bracket_token_rows: list[dict[str, Any]] = []
    tm_resolved: Path | None = None
    extract_parent_min_dt, extract_slug_min_dt = _read_extract_min_end_date_maps(extract_sort_csv)
    if not extract_sort_csv.is_file():
        print(f"warning: extract sort CSV not found (parent order falls back to Gamma endDate): {extract_sort_csv}")

    def _bracket_pkey_order(pk: str) -> tuple:
        exp = expected_by_parent.get(pk, set())
        pmr = parent_market_row_by_key.get(pk, {})
        gamma_end = str(pmr.get("market_end_date_intended") or "").strip()
        dt = _bracket_parent_sort_dt(pk, exp, extract_parent_min_dt, extract_slug_min_dt, gamma_end)
        if dt is None:
            return (1, pk)
        return (0, dt, pk)

    sorted_bracket_pkeys = sorted(expected_by_parent.keys(), key=_bracket_pkey_order)

    if token_map_path is not None and not args.no_token_map:
        tm_resolved = Path(token_map_path)
        if not tm_resolved.is_absolute():
            tm_resolved = _UTIL_ROOT / tm_resolved
        if tm_resolved.is_file():
            market_token_by_slug = _read_market_token_map(tm_resolved)
            if not market_token_by_slug:
                print(f"warning: token map read empty: {tm_resolved}")
            for pkey in sorted_bracket_pkeys:
                meta = parent_meta.get(pkey, {})
                observed_slugs_parent = observed_by_parent.get(pkey, set())
                pmr = parent_market_row_by_key.get(pkey, {})
                gamma_end = str(pmr.get("market_end_date_intended") or "").strip()
                sort_dt = _bracket_parent_sort_dt(
                    pkey,
                    expected_by_parent.get(pkey, set()),
                    extract_parent_min_dt,
                    extract_slug_min_dt,
                    gamma_end,
                )
                parent_sort_date = sort_dt.isoformat() if sort_dt else ""
                ev_ch = event_child_by_parent_slug.get(pkey, {})

                def _child_order(ms: str) -> tuple[str, str, str]:
                    ch = ev_ch.get(ms, {})
                    return (
                        str(ch.get("group_item_threshold") or ""),
                        str(ch.get("group_item_title") or ""),
                        ms,
                    )

                for mslug in sorted(expected_by_parent[pkey], key=_child_order):
                    ch = ev_ch.get(mslug, {})
                    git = str(ch.get("group_item_title") or "")
                    gith = str(ch.get("group_item_threshold") or "")
                    closed_s = str(ch.get("closed") or "")
                    end_date_s = str(ch.get("end_date") or "")
                    created_s = str(ch.get("created_at") or "")
                    fetched_s = str(ch.get("_fetched_at") or "")
                    tids = market_token_by_slug.get(mslug, [])
                    in_obs = "1" if mslug in observed_slugs_parent else "0"
                    if not tids:
                        bracket_token_rows.append(
                            {
                                "parent_key": pkey,
                                "parent_event_id": meta.get("parent_event_id", ""),
                                "parent_event_slug": meta.get("parent_event_slug", ""),
                                "parent_event_title": meta.get("parent_event_title", ""),
                                "parent_series_slug": meta.get("parent_series_slug", ""),
                                "parent_sort_date": parent_sort_date,
                                "market_slug": mslug,
                                "group_item_title": git,
                                "group_item_threshold": gith,
                                "closed": closed_s,
                                "end_date": end_date_s,
                                "created_at": created_s,
                                "_fetched_at": fetched_s,
                                "token_id": "",
                                "in_observed_export": in_obs,
                                "token_map_status": "missing_slug_in_map",
                            }
                        )
                        continue
                    for tid in tids:
                        bracket_token_rows.append(
                            {
                                "parent_key": pkey,
                                "parent_event_id": meta.get("parent_event_id", ""),
                                "parent_event_slug": meta.get("parent_event_slug", ""),
                                "parent_event_title": meta.get("parent_event_title", ""),
                                "parent_series_slug": meta.get("parent_series_slug", ""),
                                "parent_sort_date": parent_sort_date,
                                "market_slug": mslug,
                                "group_item_title": git,
                                "group_item_threshold": gith,
                                "closed": closed_s,
                                "end_date": end_date_s,
                                "created_at": created_s,
                                "_fetched_at": fetched_s,
                                "token_id": tid,
                                "in_observed_export": in_obs,
                                "token_map_status": "ok",
                            }
                        )
        else:
            print(f"warning: token map not found, skipped bracket token detail: {tm_resolved}")

    # Market-level mismatch checks
    for slug, m in market_info.items():
        pkey = m.get("parent_key") or ""
        if not pkey:
            continue
        ev_child = event_child_by_parent_slug.get(pkey, {}).get(slug)
        if ev_child is None:
            market_mismatch_rows.append(
                {
                    "slug": slug,
                    "parent_key": pkey,
                    "mismatch_type": "slug_not_in_event_markets",
                    "csv_or_market_value": slug,
                    "event_value": "",
                }
            )
            continue

        checks = [
            ("market_id", m.get("market_id", ""), ev_child.get("market_id", "")),
            ("condition_id", m.get("condition_id", ""), ev_child.get("condition_id", "")),
            ("end_date", m.get("end_date", ""), ev_child.get("end_date", "")),
            ("resolution_source", m.get("resolution_source", ""), ev_child.get("resolution_source", "")),
            ("group_item_title", m.get("group_item_title", ""), ev_child.get("group_item_title", "")),
            ("group_item_threshold", m.get("group_item_threshold", ""), ev_child.get("group_item_threshold", "")),
        ]
        for name, a, b in checks:
            if (a or "") != (b or ""):
                market_mismatch_rows.append(
                    {
                        "slug": slug,
                        "parent_key": pkey,
                        "mismatch_type": name,
                        "csv_or_market_value": a,
                        "event_value": b,
                    }
                )

    # Coverage summary
    resolved_parent_count = len([s for s in observed_slugs if (market_info.get(s, {}).get("parent_key") or "")])
    unresolved_parent_count = len(observed_slugs) - resolved_parent_count
    full_events = sum(1 for r in completeness_rows if float(r["completeness_ratio"]) == 1.0 and not r["extra_slugs"])
    incomplete_events = len(completeness_rows) - full_events
    summary = {
        "input_csv": str(input_csv),
        "observed_market_slugs": len(observed_slugs),
        "resolved_parent_linkage_n": resolved_parent_count,
        "resolved_parent_linkage_pct": round((resolved_parent_count / len(observed_slugs)) * 100.0, 4),
        "unresolved_parent_linkage_n": unresolved_parent_count,
        "events_evaluated_n": len(completeness_rows),
        "events_full_completeness_n": full_events,
        "events_incomplete_n": incomplete_events,
        "parent_markets_n": len(parent_market_rows),
        "market_mismatches_n": len(market_mismatch_rows),
        "unresolved_rows_n": len(unresolved_rows),
        "bracket_token_rows_n": len(bracket_token_rows),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    p_completeness = out_dir / f"{_STEP02_PREFIX}_event_completeness.csv"
    p_mismatch = out_dir / f"{_STEP02_PREFIX}_market_mismatches.csv"
    p_unresolved = out_dir / f"{_STEP02_PREFIX}_unresolved_markets.csv"
    p_parent_markets = out_dir / f"{_STEP02_PREFIX}_parent_markets.csv"
    p_bracket_tokens = out_dir / f"{_STEP02_PREFIX}_bracket_market_tokens.csv"
    p_summary = out_dir / "coverage_summary.json"

    _write_csv(
        p_completeness,
        completeness_rows,
        [
            "parent_key",
            "parent_event_id",
            "parent_event_slug",
            "parent_event_title",
            "parent_series_slug",
            "expected_n_markets",
            "observed_n_markets",
            "matched_n_markets",
            "completeness_ratio",
            "missing_slugs",
            "extra_slugs",
            "unique_end_dates_in_event",
            "unique_resolution_sources_in_event",
            "duplicate_group_item_thresholds",
            "duplicate_group_item_titles",
        ],
    )
    if tm_resolved is not None and tm_resolved.is_file():
        _write_csv(
            p_bracket_tokens,
            bracket_token_rows,
            [
                "parent_key",
                "parent_event_id",
                "parent_event_slug",
                "parent_event_title",
                "parent_series_slug",
                "parent_sort_date",
                "market_slug",
                "group_item_title",
                "group_item_threshold",
                "closed",
                "end_date",
                "created_at",
                "_fetched_at",
                "token_id",
                "in_observed_export",
                "token_map_status",
            ],
        )

    _write_csv(
        p_mismatch,
        market_mismatch_rows,
        ["slug", "parent_key", "mismatch_type", "csv_or_market_value", "event_value"],
    )
    _write_csv(
        p_unresolved,
        unresolved_rows,
        ["slug", "stage", "reason", "parent_event_id", "parent_event_slug"],
    )
    _write_csv(
        p_parent_markets,
        sorted(
            parent_market_rows,
            key=lambda x: (
                # Put rows with missing intended end date at the bottom.
                1 if not str(x.get("market_end_date_intended") or "").strip() else 0,
                str(x.get("market_end_date_intended") or ""),
                str(x.get("parent_market_slug") or x.get("meeting_key") or ""),
            ),
        ),
        [
            "parent_market_id",
            "parent_market_slug",
            "meeting_title",
            "meeting_series_slug",
            "meeting_key",
            "market_end_date_intended",
            "market_end_date_actual_resolution",
            "event_status",
            "expected_n_markets",
            "observed_n_markets",
            "missing_market_slugs",
            "extra_market_slugs",
        ],
    )
    p_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if printed_progress_dot:
        print()
    print(f"wrote: {p_completeness}")
    if tm_resolved is not None and tm_resolved.is_file():
        print(f"wrote: {p_bracket_tokens}")
    print(f"wrote: {p_mismatch}")
    print(f"wrote: {p_unresolved}")
    print(f"wrote: {p_parent_markets}")
    print(f"wrote: {p_summary}")
    print(f"summary: {json.dumps(summary)}")
    if incomplete_events > 0 and not args.allow_incomplete:
        print(
            "Coverage check failed: incomplete events found. "
            f"events_incomplete_n={incomplete_events}. "
            "Use --allow-incomplete to return success anyway."
        )
        return 7
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
