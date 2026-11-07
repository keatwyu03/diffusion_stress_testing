"""
Saves everything needed to evaluate one finished rare-event (Doob-h) channel
later -- WITHOUT changing the evaluation itself: generated windows, the real
event windows, their timestamps, the conditioning (event) values, and every
input of the inverse transform back to daily log returns.

It rebuilds the SAME DataProcessor, the SAME event mask and the SAME seeded
draw of entry windows as doobh_evaluation.py / doobh_evaluation_exp2.py, so the
saved arrays line up with what the evaluator compares.

Inverse transform per generated window i:
    model units (per-window EMA z)  --entry stats of real event window pick[i]-->  CSV units
    inclusion : CSV units are raw daily log returns (done)
    embedding : CSV units are method k's NN residuals z^(k); returns
                r = mu_k(v_t) + sigma_k(v_t) * z on the dates of window pick[i]
                (mu/sigma from that method's own fitted standardizer, standardize_own_macro.py)

Writes <results>/<channel>/rare_event_artifacts.npz + rare_event_artifacts.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from config import get_default_config  # noqa: E402
from data.data_processor import DataProcessor  # noqa: E402
from doobh_evaluation import window_z_start_end  # noqa: E402

METHOD_DIR = {"exp2_s0": "S0_state_space", "exp2_s1": "S1_linear", "exp2_s2": "S2_forward_fill",
              "exp2_s3": "S3_cubic_spline", "exp2_s4": "S4_tracking_regression"}


def build_processor(csv_path, cfg):
    dp = DataProcessor(
        csv_path=csv_path, tickers=cfg.data.tickers, weekday_col=cfg.data.weekday_col,
        seq_len=cfg.data.seq_len, test_days=cfg.data.test_days, start_date=cfg.data.start_date,
        end_date=cfg.data.end_date, train_end_date=cfg.data.train_end_date,
        window_shift=cfg.data.window_shift, winsorize_lower=cfg.data.winsorize_lower,
        winsorize_upper=cfg.data.winsorize_upper, ema_span=cfg.data.ema_span,
        event_causal=cfg.data.event_causal, use_ema_standardization=cfg.data.use_ema_standardization,
        event_lag_gap=cfg.data.event_lag_gap)
    dp.process_all()
    return dp


def window_dates(dp, idx, T):
    d = dp.df.index
    return (np.array([str(d[j].date()) for j in idx]), np.array([str(d[j + T - 1].date()) for j in idx]))


def macro_at(table: pd.DataFrame, dp, idx, T, tickers):
    """(n, A, T) values of a Date x ticker table on the days of each window j in idx."""
    vals = table.reindex(dp.df.index)[tickers].to_numpy(np.float64)
    return np.stack([vals[j:j + T].T for j in idx])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", required=True, choices=["inclusion", "embedding"])
    p.add_argument("--channel", required=True)
    p.add_argument("--results-root", required=True, help="folder under baseline_experiments/main_results/")
    p.add_argument("--csv-suffix", default="")
    p.add_argument("--standardization-dir", default=None, help="embedding: folder with <method>/mu.csv, sigma.csv")
    p.add_argument("--out-dir", default=None, help="default: the channel's results folder")
    args = p.parse_args()

    cfg = get_default_config()
    T = cfg.data.seq_len
    tickers = list(cfg.data.tickers)
    csv_path = os.path.join(_ROOT, "explore", "standardization_results",
                            f"data_for_diffusion_{args.channel}{args.csv_suffix}.csv")
    res_dir = os.path.join(os.path.dirname(_HERE), "main_results", args.results_root, args.channel)
    out_dir = args.out_dir or res_dir
    dp = build_processor(csv_path, cfg)

    if args.group == "inclusion":
        import doobh_evaluation as ev
        valid = ev.common_valid_mask(dp, cfg, "train")
        mask, threshold, _ = ev.build_event_mask(args.channel, dp, cfg, "train", valid)
    else:
        import doobh_evaluation_exp2 as ev
        ev.CSV_SUFFIX = args.csv_suffix
        valid = ev.self_valid_mask(args.channel, dp, cfg, "train")
        mask, threshold, _ = ev.build_event_mask(args.channel, dp, cfg, "train", valid)
    n_train = dp.X_train.shape[0]
    values = ev.channel_macro_values(args.channel, dp.df.index)
    zs_all, ze_all = window_z_start_end(values, T, n_train + dp.X_test.shape[0], cfg.data.event_causal, cfg.data.event_lag_gap)
    zs, ze = zs_all[:n_train], ze_all[:n_train]

    # real event windows (train split, in-sample) -- same as the evaluator
    real_idx = np.flatnonzero(mask)
    real_units = dp.destandardize_windows(np.transpose(dp.X_train.numpy()[mask], (0, 2, 1)),
                                          dp.mu_entry_train[mask], dp.sig_entry_train[mask])
    align = np.stack([dp.df[tickers].to_numpy(np.float64)[j:j + T].T for j in real_idx])
    align_err = float(np.max(np.abs(real_units - align)))
    if align_err > 1e-4:
        raise RuntimeError(f"window index -> date alignment failed (max err {align_err})")

    # generated windows + the evaluator's seeded entry-window draw (sample_entry_stats, reproduced to get indices)
    gen_z = torch.load(os.path.join(res_dir, "generated_samples_test.pt"), map_location="cpu", weights_only=False)
    gen_z = gen_z.numpy() if hasattr(gen_z, "numpy") else np.asarray(gen_z)
    n_gen = len(gen_z)
    pick = real_idx[np.random.default_rng(cfg.seed).integers(0, len(real_idx), size=n_gen)]
    mu_e, sig_e = dp.sample_entry_stats(n_draw=n_gen, split="train", mask=mask, seed=cfg.seed)
    if not (np.allclose(mu_e, dp.mu_entry_train[pick]) and np.allclose(sig_e, dp.sig_entry_train[pick])):
        raise RuntimeError("could not reproduce the evaluator's entry-window draw")
    gen_units = dp.destandardize_windows(gen_z, mu_e, sig_e)

    raw = pd.read_csv(os.path.join(_ROOT, "explore", "macro_data_new.csv"), parse_dates=["Date"]).set_index("Date")
    real_raw = macro_at(raw, dp, real_idx, T, tickers)
    rs, re_ = window_dates(dp, real_idx, T)
    gs, ge = window_dates(dp, pick, T)
    out = dict(gen_model_units=gen_z.astype(np.float32), gen_entry_mu=mu_e, gen_entry_sigma=sig_e,
               gen_window_index=pick, gen_window_start=gs, gen_window_end=ge, gen_csv_units=gen_units.astype(np.float32),
               real_window_index=real_idx, real_window_start=rs, real_window_end=re_,
               real_csv_units=real_units.astype(np.float32), real_returns_raw=real_raw.astype(np.float32),
               real_z_start=zs[mask], real_z_end=ze[mask], real_event_metric=(ze - zs)[mask], event_threshold=threshold)
    meta = {"group": args.group, "channel": args.channel, "channel_csv": csv_path, "split": "train (in-sample)",
            "event_type": cfg.hfunction.event_type, "event_top_fraction": cfg.hfunction.event_threshold,
            "event_causal": cfg.data.event_causal, "event_lag_gap": cfg.data.event_lag_gap,
            "n_real_events": int(len(real_idx)), "n_generated": int(n_gen), "entry_draw_seed": cfg.seed,
            "window_index": "row j of DataProcessor.df -> window covers rows j..j+seq_len-1",
            "winsorization": f"CSV units are winsorized at train quantiles [{cfg.data.winsorize_lower}, {cfg.data.winsorize_upper}] "
                             "inside DataProcessor; real_returns_raw is unwinsorized",
            "alignment_check_max_abs_err": align_err}
    if args.group == "inclusion":
        out["gen_returns"] = gen_units.astype(np.float32)
        meta["units"] = "CSV units = raw daily log returns; gen_returns = gen_csv_units"
    else:
        sdir = os.path.join(args.standardization_dir, METHOD_DIR[args.channel])
        mu_t = pd.read_csv(os.path.join(sdir, "mu.csv"), parse_dates=["Date"]).set_index("Date")
        sg_t = pd.read_csv(os.path.join(sdir, "sigma.csv"), parse_dates=["Date"]).set_index("Date")
        g_mu, g_sg = macro_at(mu_t, dp, pick, T, tickers), macro_at(sg_t, dp, pick, T, tickers)
        r_mu, r_sg = macro_at(mu_t, dp, real_idx, T, tickers), macro_at(sg_t, dp, real_idx, T, tickers)
        for name, a in [("gen mu", g_mu), ("gen sigma", g_sg), ("real mu", r_mu), ("real sigma", r_sg)]:
            if not np.isfinite(a).all():
                raise RuntimeError(f"{name}: non-finite standardizer values on window dates")
        real_rec = r_mu + r_sg * real_units
        out.update(gen_returns=(g_mu + g_sg * gen_units).astype(np.float32), gen_macro_mu=g_mu.astype(np.float32),
                   gen_macro_sigma=g_sg.astype(np.float32), real_macro_mu=r_mu.astype(np.float32),
                   real_macro_sigma=r_sg.astype(np.float32), real_returns_reconstructed=real_rec.astype(np.float32))
        exact = np.abs(real_rec - real_raw) < 1e-6
        meta.update(units=f"CSV units = {args.channel} NN residuals z^(k); gen_returns = mu_k(v_t) + sigma_k(v_t) * gen_csv_units "
                          "on the dates of the paired real event window",
                    standardizer=sdir, real_reconstruction_exact_fraction=float(exact.mean()),
                    real_reconstruction_note="inexact entries are exactly the winsorized ones")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "rare_event_artifacts.npz"), **out)
    with open(os.path.join(out_dir, "rare_event_artifacts.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{args.channel}] saved rare_event_artifacts ({len(real_idx)} real events, {n_gen} generated) -> {out_dir}")


if __name__ == "__main__":
    main()
