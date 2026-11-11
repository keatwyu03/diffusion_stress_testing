"""Locked macro inputs: restore the frozen Table 4 panels (git a6c4f3f, 2026-08-23) into the live
latent_state_estimation/*.csv files (downloading is disabled; this snapshot is the only macro source).

Data window (config.data.start_date / end_date): the DAILY panels are restricted to the window (line filter on
the date column, so every kept value is byte-identical to the snapshot). The MONTHLY panels are restored exactly
as the Table 4 run had them: the monthly PCA calibration (monthly_first_pc) uses their full history, so cutting
them would change m_t and Table 4 would no longer reproduce. The state-space model itself only uses monthly
observations inside the daily window."""
import os
import shutil

MONTHLY = ["growth_macro", "inflation_macro", "vol_macro"]
DAILY = ["growth_daily", "inflation_daily", "vol_daily"]
PANELS = MONTHLY + DAILY
_HERE = os.path.dirname(os.path.abspath(__file__))


def _copy_daily_in_window(src, dst, start_date, end_date):
    lo = str(start_date)[:10] if start_date else None
    hi = str(end_date)[:10] if end_date else None
    kept = 0
    with open(src) as f, open(dst, "w") as g:
        g.write(f.readline())  # header
        for line in f:
            d = line[:10]       # ISO date YYYY-MM-DD at the start of each row
            if (lo is None or d >= lo) and (hi is None or d <= hi):
                g.write(line)
                kept += 1
    return kept


def restore_frozen_panels(snapshot_dir: str, start_date=None, end_date=None) -> None:
    src = snapshot_dir if os.path.isabs(snapshot_dir) else os.path.join(os.path.dirname(_HERE), snapshot_dir)
    for p in MONTHLY:
        shutil.copy2(os.path.join(src, f"{p}.csv"), os.path.join(_HERE, f"{p}.csv"))
    for p in DAILY:
        n = _copy_daily_in_window(os.path.join(src, f"{p}.csv"), os.path.join(_HERE, f"{p}.csv"), start_date, end_date)
        print(f"[frozen macro data] {p}: {n} days in window {start_date}..{end_date}")
    print(f"[frozen macro data] restored {len(PANELS)} panels from {src} (no download; monthly panels unmodified)")
