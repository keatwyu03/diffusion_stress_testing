"""
Sliding-window analysis: find the data window where train-set and test-set
event portfolio returns are most similar across three strategies.

Event condition : TSLA last EVENT_WINDOW days raw log return sum <= EVENT_THRESHOLD_RAW
Portfolio metric: last LAST_DAYS-day cumulative return for each of:
                  equal-weight / min-variance / risk-parity
Score           : -(|ew_diff| + |mv_diff| + |rp_diff|) / 3  (higher = better)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH            = "Stocks_logret.csv"
TICKERS             = ["AAPL", "AMZN", "JPM", "TSLA"]
WEEKDAY_COL         = "weekday"
SEQ_LEN             = 64
WINDOW_FOR_COV      = 54    # first 54 days used to estimate covariance
EVENT_ASSET_IDX     = 3     # TSLA
EVENT_WINDOW        = 10    # last 10 days of each 64-day window
EVENT_THRESHOLD_RAW = -0.05 # TSLA last-10-day sum ≤ -5%
LAST_DAYS           = 5     # last 5 days for portfolio cumulative return
MIN_EVENTS          = 3     # minimum events needed in each split to score
TRAIN_RATIO         = 0.75
WIN_LENGTHS         = range(2000, 3001, 50)
SLIDE_STEP          = 50
RESULTS_DIR         = "results/regime"
os.makedirs(RESULTS_DIR, exist_ok=True)

N_ASSETS = len(TICKERS)

# ── Portfolio weight solvers ───────────────────────────────────────────────────
def _min_var_weights(cov: np.ndarray) -> np.ndarray:
    w0 = np.ones(N_ASSETS) / N_ASSETS
    res = minimize(
        lambda w: float(w @ cov @ w),
        w0,
        jac=lambda w: 2 * cov @ w,
        method="SLSQP",
        bounds=[(0, 1)] * N_ASSETS,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        options={"ftol": 1e-9, "maxiter": 300},
    )
    return res.x if res.success else w0


def _risk_parity_weights(cov: np.ndarray) -> np.ndarray:
    w0 = np.ones(N_ASSETS) / N_ASSETS

    def obj(w):
        pv = float(w @ cov @ w)
        if pv < 1e-12:
            return 0.0
        rc = w * (cov @ w) / pv
        return sum((rc[i] - rc[j]) ** 2 for i in range(N_ASSETS) for j in range(i + 1, N_ASSETS))

    res = minimize(
        obj,
        w0,
        method="SLSQP",
        bounds=[(1e-6, 1)] * N_ASSETS,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        options={"ftol": 1e-9, "maxiter": 300},
    )
    return res.x if res.success else w0


# ── 1. Load data ───────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").set_index("Date")
df = df[TICKERS + [WEEKDAY_COL]].dropna()
N_TOTAL  = len(df)
dates    = df.index
raw_vals = df[TICKERS].values  # (N_TOTAL, 4)

print(f"Data: {dates[0].date()} ~ {dates[-1].date()}  ({N_TOTAL} days)")

# ── 2. Pre-compute per-sequence: event flag + three portfolio returns ──────────
# Sequence ending at index t: days [t-SEQ_LEN+1 .. t]
#   cov  : first WINDOW_FOR_COV days  → indices [t-63 .. t-63+54) = [t-63 .. t-9)
#   last5: last LAST_DAYS days         → indices [t-4  .. t]
#   event: TSLA last EVENT_WINDOW days → indices [t-9  .. t]
print(f"\nPre-computing per-sequence portfolio returns ({N_TOTAL - SEQ_LEN + 1} sequences)...")

seq_records = []
w_ew = np.ones(N_ASSETS) / N_ASSETS

for t in range(SEQ_LEN - 1, N_TOTAL):
    tsla_sum = raw_vals[t - EVENT_WINDOW + 1 : t + 1, EVENT_ASSET_IDX].sum()
    is_event = tsla_sum <= EVENT_THRESHOLD_RAW

    last5 = raw_vals[t - LAST_DAYS + 1 : t + 1, :]  # (5, 4)

    ew_ret = float((last5 @ w_ew).sum())

    if is_event:
        cov_data = raw_vals[t - SEQ_LEN + 1 : t - SEQ_LEN + 1 + WINDOW_FOR_COV, :]
        cov = np.cov(cov_data.T) + 1e-6 * np.eye(N_ASSETS)
        w_mv = _min_var_weights(cov)
        w_rp = _risk_parity_weights(cov)
        mv_ret = float((last5 @ w_mv).sum())
        rp_ret = float((last5 @ w_rp).sum())
    else:
        mv_ret = np.nan
        rp_ret = np.nan

    seq_records.append((dates[t], is_event, ew_ret, mv_ret, rp_ret))

    if (t - SEQ_LEN + 1) % 500 == 0:
        print(f"  {t - SEQ_LEN + 1}/{N_TOTAL - SEQ_LEN + 1}", flush=True)

seq_df = pd.DataFrame(seq_records, columns=["date", "event", "ew_ret", "mv_ret", "rp_ret"])
seq_df = seq_df.set_index("date")

ev_df = seq_df[seq_df["event"]]
total_events = len(ev_df)
print(f"Total event sequences: {total_events}")
print(f"  EW  mean ret: {ev_df['ew_ret'].mean():.5f}")
print(f"  MV  mean ret: {ev_df['mv_ret'].mean():.5f}")
print(f"  RP  mean ret: {ev_df['rp_ret'].mean():.5f}")

# Rolling vol for context plot
roll_vol_mean = df[TICKERS].rolling(60).std().mean(axis=1)


# ── 3. Helper: mean portfolio returns for event sequences in a slice ───────────
def slice_stats(seq_slice: pd.DataFrame):
    ev = seq_slice[seq_slice["event"]]
    n = len(ev)
    if n == 0:
        return n, np.nan, np.nan, np.nan
    return n, float(ev["ew_ret"].mean()), float(ev["mv_ret"].mean()), float(ev["rp_ret"].mean())


# ── 4. Sliding-window search ───────────────────────────────────────────────────
print(f"\nSearching {len(list(WIN_LENGTHS))} lengths × step={SLIDE_STEP}...")
results = []

for win_len in WIN_LENGTHS:
    for start in range(0, N_TOTAL - win_len + 1, SLIDE_STEP):
        end        = start + win_len
        win_start  = dates[start]
        win_end    = dates[end - 1]
        split_idx  = start + int(win_len * TRAIN_RATIO)
        train_end  = dates[split_idx - 1]
        test_start = dates[split_idx]

        train_slice = seq_df.loc[win_start:train_end]
        test_slice  = seq_df.loc[test_start:win_end]

        n_tr, tr_ew, tr_mv, tr_rp = slice_stats(train_slice)
        n_te, te_ew, te_mv, te_rp = slice_stats(test_slice)

        if n_tr < MIN_EVENTS or n_te < MIN_EVENTS:
            continue
        if any(np.isnan(v) for v in [tr_ew, tr_mv, tr_rp, te_ew, te_mv, te_rp]):
            continue

        ew_diff = abs(tr_ew - te_ew)
        mv_diff = abs(tr_mv - te_mv)
        rp_diff = abs(tr_rp - te_rp)
        score   = -(ew_diff + mv_diff + rp_diff) / 3.0

        results.append({
            "win_len":    win_len,
            "start":      str(win_start.date()),
            "end":        str(win_end.date()),
            "train_end":  str(train_end.date()),
            "test_start": str(test_start.date()),
            "n_train":    len(train_slice),
            "n_test":     len(test_slice),
            "n_train_ev": n_tr,
            "n_test_ev":  n_te,
            "tr_ew": round(tr_ew, 6), "te_ew": round(te_ew, 6), "ew_diff": round(ew_diff, 6),
            "tr_mv": round(tr_mv, 6), "te_mv": round(te_mv, 6), "mv_diff": round(mv_diff, 6),
            "tr_rp": round(tr_rp, 6), "te_rp": round(te_rp, 6), "rp_diff": round(rp_diff, 6),
            "score":  round(score, 6),
        })

res_df = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)

# ── 5. All-data baseline ───────────────────────────────────────────────────────
all_split    = int(N_TOTAL * TRAIN_RATIO)
all_train_end  = dates[all_split - 1]
all_test_start = dates[all_split]
n_tr_a, tr_ew_a, tr_mv_a, tr_rp_a = slice_stats(seq_df.loc[:all_train_end])
n_te_a, te_ew_a, te_mv_a, te_rp_a = slice_stats(seq_df.loc[all_test_start:])
ew_d_a = abs(tr_ew_a - te_ew_a)
mv_d_a = abs(tr_mv_a - te_mv_a)
rp_d_a = abs(tr_rp_a - te_rp_a)
all_score = -(ew_d_a + mv_d_a + rp_d_a) / 3.0
all_row = {
    "win_len": N_TOTAL, "start": str(dates[0].date()), "end": str(dates[-1].date()),
    "train_end": str(all_train_end.date()), "test_start": str(all_test_start.date()),
    "n_train": all_split, "n_test": N_TOTAL - all_split,
    "n_train_ev": n_tr_a, "n_test_ev": n_te_a,
    "tr_ew": round(tr_ew_a, 6), "te_ew": round(te_ew_a, 6), "ew_diff": round(ew_d_a, 6),
    "tr_mv": round(tr_mv_a, 6), "te_mv": round(te_mv_a, 6), "mv_diff": round(mv_d_a, 6),
    "tr_rp": round(tr_rp_a, 6), "te_rp": round(te_rp_a, 6), "rp_diff": round(rp_d_a, 6),
    "score": round(all_score, 6),
}

# ── 6. Save CSV ────────────────────────────────────────────────────────────────
csv_path = os.path.join(RESULTS_DIR, "sliding_window_splits.csv")
res_df.to_csv(csv_path, index=False)
print(f"\nSaved {len(res_df)} candidates to {csv_path}")

# ── 7. Print results ───────────────────────────────────────────────────────────
cols_short = ["win_len", "start", "end", "train_end",
              "n_train_ev", "n_test_ev",
              "ew_diff", "mv_diff", "rp_diff", "score"]

print("\n" + "=" * 80)
print("TOP 15 CANDIDATES")
print("=" * 80)
print(res_df.head(15)[cols_short].to_string(index=True))

print("\n" + "=" * 80)
print("BEST PER WINDOW LENGTH")
print("=" * 80)
print(res_df.groupby("win_len").first().reset_index()[cols_short].to_string(index=False))

print("\n" + "=" * 80)
print("BASELINE: ALL DATA")
print("=" * 80)
print(pd.DataFrame([all_row])[cols_short].to_string(index=False))
rank = int((res_df["score"] > all_row["score"]).sum()) + 1
print(f"\n  All-data ranks #{rank} / {len(res_df) + 1}")

# ── 8. Plot ────────────────────────────────────────────────────────────────────
top3 = res_df.head(3)
win_colors = ["#1B5E20", "#E65100", "#1A237E"]

fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

axes[0].plot(roll_vol_mean.index, roll_vol_mean.values, color="black", lw=0.8)
axes[0].set_ylabel("Mean Vol")
axes[0].set_title("Rolling 60-day Volatility")

ev_seq = seq_df[seq_df["event"]]
axes[1].scatter(ev_seq.index, ev_seq["ew_ret"], s=6, color="C0", alpha=0.4, label="EW")
axes[1].scatter(ev_seq.index, ev_seq["mv_ret"], s=6, color="C1", alpha=0.4, label="MV")
axes[1].scatter(ev_seq.index, ev_seq["rp_ret"], s=6, color="C2", alpha=0.4, label="RP")
axes[1].axhline(0, color="gray", lw=0.8, ls="--")
axes[1].set_ylabel("Port Ret (last 5d)")
axes[1].set_title(f"Event Sequences: Last-{LAST_DAYS}-Day Portfolio Returns")
axes[1].legend(fontsize=8)

for i, (_, row) in enumerate(top3.iterrows()):
    ws = pd.to_datetime(row["start"])
    te = pd.to_datetime(row["train_end"])
    we = pd.to_datetime(row["end"])
    for ax in axes:
        ax.axvspan(ws, te, alpha=0.10, color=win_colors[i])
        ax.axvspan(te, we, alpha=0.18, color=win_colors[i])
        ax.axvline(te, color=win_colors[i], lw=1.2, ls="--")
    label = (f"#{i+1} {row['start']}~{row['end']}  "
             f"ew={row['ew_diff']:.4f} mv={row['mv_diff']:.4f} rp={row['rp_diff']:.4f}")
    axes[0].axvline(ws, color=win_colors[i], lw=1, ls=":", label=label)

axes[0].legend(fontsize=7, loc="upper left")
plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, "sliding_window_analysis.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to {plot_path}")

# ── 9. Recommendation ─────────────────────────────────────────────────────────
def print_row(label, row):
    print(f"\n{label}  Window: {row['start']} ~ {row['end']}  (length={row['win_len']} days)")
    print(f"      Train: {row['start']} ~ {row['train_end']}  ({row['n_train']} seqs, {row['n_train_ev']} events)")
    print(f"      Test:  {row['test_start']} ~ {row['end']}  ({row['n_test']} seqs, {row['n_test_ev']} events)")
    print(f"      EW  train={row['tr_ew']:.5f}  test={row['te_ew']:.5f}  |diff|={row['ew_diff']:.5f}")
    print(f"      MV  train={row['tr_mv']:.5f}  test={row['te_mv']:.5f}  |diff|={row['mv_diff']:.5f}")
    print(f"      RP  train={row['tr_rp']:.5f}  test={row['te_rp']:.5f}  |diff|={row['rp_diff']:.5f}")
    print(f"      Score (avg diff, negated): {row['score']:.5f}")

print("\n" + "=" * 80)
print("RECOMMENDATION (Top 3)")
print("=" * 80)
for i, (_, row) in enumerate(top3.iterrows()):
    print_row(f"[#{i+1}]", row)

print("\n" + "=" * 80)
print("BASELINE: ALL DATA")
print("=" * 80)
print_row("[ALL]", all_row)
print(f"\n  All-data ranks #{rank} / {len(res_df) + 1}")
