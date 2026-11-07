"""
Re-analysis of already-generated S0 vs S2 samples, stratified by "trading
days since the last monthly macro print" at each test window's conditioning
date. No retraining/resampling -- reuses saved generated_samples.npy/
test_data.npz and recomputes energy_score/marginal-W1 within staleness
buckets instead of one blended average, to check whether S0's (SSM) value
over S2 (forward-fill) is concentrated where forward-fill is structurally
blind (late in the month, before the next print) and absent near a fresh
print (where forward-fill isn't stale yet).
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASELINE = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _BASELINE)

import common  # noqa: E402
from daily_macro_states import active_groups, load_monthly_panel  # noqa: E402
from diffusion_adapters import build_cond_feature, per_window_cond_values, common_valid_window_mask  # noqa: E402
from tracking_regression import monthly_first_pc  # noqa: E402
from exp_evaluation import energy_score  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(_HERE), "main_results", "exp2_results")


def main():
    cfg = common.get_default_config()
    manifest = common.get_window_manifest(cfg)
    seq_len = manifest["seq_len"]

    # Reconstruct exactly which test windows were used (same mask every stage computed)
    methods = common.METHODS
    all_cond = {m: per_window_cond_values(build_cond_feature(m, manifest, seq_len), manifest) for m in methods}
    mask = common_valid_window_mask(all_cond)
    test_start_idx = manifest["test_start_idx"]
    test_mask = mask[test_start_idx:]
    test_starts = manifest["starts"][test_start_idx:][test_mask]
    cond_dates = manifest["df"].index[test_starts - 1]  # date each window's conditioning was observed
    trading_dates = manifest["df"].index

    # Union of growth+inflation monthly print (anchor) dates, same convention as S1-S3/S4
    groups, sel = active_groups(cfg)
    anchor_dates = sorted(set().union(*[
        set(common.month_end_anchor(monthly_first_pc(load_monthly_panel(g, sel[g]))[0], trading_dates).index)
        for g in groups
    ]))
    anchor_dates = pd.DatetimeIndex(anchor_dates)
    date_pos = pd.Series(np.arange(len(trading_dates)), index=trading_dates)

    staleness = []
    for d in cond_dates:
        prior_anchors = anchor_dates[anchor_dates <= d]
        if len(prior_anchors) == 0:
            staleness.append(np.nan)
            continue
        last_anchor = prior_anchors.max()
        staleness.append(int(date_pos[d] - date_pos[last_anchor]))
    staleness = np.array(staleness)
    valid = ~np.isnan(staleness)
    print(f"Test windows: {len(staleness)}, {valid.sum()} with a prior anchor "
          f"(staleness range {int(np.nanmin(staleness))}-{int(np.nanmax(staleness))} trading days)")

    bins = [(0, 5), (6, 10), (11, 15), (16, 30)]
    bucket_idx = np.full(len(staleness), -1)
    for bi, (lo, hi) in enumerate(bins):
        bucket_idx[(staleness >= lo) & (staleness <= hi)] = bi

    real = np.load(os.path.join(OUT_DIR, common.METHOD_DIRNAMES["s0"], "diffusion", "test_data.npz"))["X_test"]
    gen = {m: np.load(os.path.join(OUT_DIR, common.METHOD_DIRNAMES[m], "diffusion", "generated_samples.npy"))
           for m in ("s0", "s2")}
    n = min(len(real), len(staleness), gen["s0"].shape[0], gen["s2"].shape[0])

    print(f"\n{'Bucket (trading days since print)':38s} {'N':>5s} {'S0 energy':>10s} {'S2 energy':>10s} "
          f"{'S0 marg.W1':>11s} {'S2 marg.W1':>11s}")
    for bi, (lo, hi) in enumerate(bins):
        idx = np.where(bucket_idx[:n] == bi)[0]
        if len(idx) == 0:
            print(f"{lo}-{hi} days: no windows")
            continue
        es = {m: np.mean([energy_score(real[i], gen[m][i]) for i in idx]) for m in ("s0", "s2")}
        mw1 = {}
        for m in ("s0", "s2"):
            A = real.shape[1]
            ws = [wasserstein_distance(real[idx, a, -1], gen[m][idx][:, :, a, -1].reshape(-1)) for a in range(A)]
            mw1[m] = np.mean(ws)
        print(f"{lo}-{hi} days{'':28s} {len(idx):5d} {es['s0']:10.4f} {es['s2']:10.4f} "
              f"{mw1['s0']:11.5f} {mw1['s2']:11.5f}")


if __name__ == "__main__":
    main()
