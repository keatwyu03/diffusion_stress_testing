"""
Preflight checks for the corrected FULL-macro runs (2026-11-07). Checks the
actual data loaders, cached files and checkpoints -- not just configs.
Exit code 0 = every check passed. Re-runnable:

    venv/bin/python3 baseline_experiments/preflight_full_v2.py

Macro inclusion (M0-M4): raw daily log-return targets everywhere, M0 unconditional.
Macro embedding (S0-S4): each method trains on its OWN NN residuals.
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

ROOT = "/scr/kadenwu/Conditional_diffusion"
EB = ROOT + "/baseline_experiments"
FULL = EB + "/experiment_1 (Full)"
STD = FULL + "/extras/standardization_own"
COPIES = {k: f"/scr/kadenwu/Conditional_diffusion_v2_rare_{k}" for k in ["inc_nc", "inc_c", "emb_nc", "emb_c"]}
METHODS = {"s0": "S0_state_space", "s1": "S1_linear", "s2": "S2_forward_fill", "s3": "S3_cubic_spline",
           "s4": "S4_tracking_regression"}
TICKERS = ["IBM", "CSCO", "AAPL", "MSFT", "ORCL", "INTC", "TXN", "QCOM", "AMAT", "ADBE", "KO", "PG", "CL", "GIS", "KMB"]
sys.path[:0] = [ROOT, EB + "/macro_inclusion", EB + "/macro_embed_method", ROOT + "/explore"]
RESULTS = []
JOBS = {"inc_nc": "18364671", "inc_c": "18364672"}  # submitted inclusion rare-event jobs


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)


RAW = pd.read_csv(ROOT + "/explore/macro_data_new.csv", parse_dates=["Date"]).set_index("Date")


def inclusion_conditional():
    d = FULL + "/causal/macro_inclusion/conditional_only"
    e = np.load(d + "/shared/eval_data.npz")
    dates = pd.DatetimeIndex(RAW.index)
    st = dates.get_indexer(pd.to_datetime(e["window_start_dates"]))
    Xr = np.stack([RAW[TICKERS].to_numpy()[s:s + 10].T for s in st])
    check("inclusion/conditional-only: eval windows are RAW log returns (running job 18359403)",
          np.allclose(Xr, e["X_eval"]), f"{len(st)} windows, split={e['eval_split']}")
    m0 = json.load(open(d + "/M0_unconditional/checkpoint_meta.json"))
    check("inclusion/conditional-only: M0 is unconditional", m0["conditioning_feature"] == {} and m0["condition_norm_stats"] is None)
    scales = {json.load(open(f))["scale_factor"] for f in glob.glob(d + "/M*/checkpoint_meta.json")}
    shared = json.load(open(d + "/shared/scale_factor.json"))["scale_factor"]
    check("inclusion/conditional-only: one shared numeric scale for every trained model", scales == {shared}, f"{shared:.4f}")
    # checkpoint target: the saved scale is 1/std of the TRAINING targets -> must equal 1/std of raw-return train windows
    import main_exp1 as m1
    cfg = m1.get_default_config(); cfg.data.event_causal = True
    df, mc = m1.load_raw_data(cfg)
    cdf = m1.compute_conditioning_variables(df, mc, 10, 10, 59 / 61, 10)
    X, *_rest, starts = m1.build_windows(df, cdf, 10, event_causal=True, event_lag_gap=1)
    split, _ = m1.train_test_split_idx(starts, _rest[2], 10, cfg)
    check("inclusion/conditional-only: checkpoints were trained on RAW returns (scale == 1/std of raw train windows)",
          abs(shared - 1 / X[:split].std()) < 1e-6 * shared, f"saved {shared:.4f}, raw {1 / X[:split].std():.4f} (residual targets would give ~1)")
    for f in sorted(glob.glob(d + "/M[1-4]*/checkpoint_meta.json")):
        cf = json.load(open(f))["conditioning_feature"]
        check(f"inclusion/conditional-only: {f.split('/')[-2]} conditioner", cf.get("conditioner") == "causal_value_window"
              and cf.get("event_causal") is True and cf.get("event_lag_gap") == 1, str(cf.get("variable")))


def inclusion_rare(key, causal):
    C = COPIES[key]
    cfgtxt = open(C + "/config/config.py").read()
    check(f"inclusion/rare {key}: copy config latent_standardized=False (raw targets, tag _EWMA)",
          "    latent_standardized: bool = False" in cfgtxt)
    # a running launcher creates these backups itself; they must be copies of the FRESH files, never stale ones
    csv_bk = glob.glob(C + "/explore/standardization_results/*orig_backup*")
    cfg_bk = glob.glob(C + "/config/*orig_backup*")
    bk_ok = all(subprocess.run(["cmp", "-s", f, ROOT + "/explore/standardization_results/data_for_diffusion.csv"]).returncode == 0
                for f in csv_bk) and all("    latent_standardized: bool = False" in open(f).read()
                                         and "    event_causal: bool = False" in open(f).read() for f in cfg_bk)
    check(f"inclusion/rare {key}: config/CSV backups (if any) are copies of the fresh files", bk_ok, f"{len(csv_bk) + len(cfg_bk)} backups")
    job = JOBS.get(key)
    start = pd.Timestamp(subprocess.run(["sacct", "-j", job, "-X", "-n", "--format=Start"], capture_output=True, text=True).stdout.strip()) if job else None
    for f in glob.glob(C + "/ckpt_new/*"):
        fresh = start is not None and pd.Timestamp.fromtimestamp(os.path.getmtime(f)) >= start
        check(f"inclusion/rare {key}: checkpoint {os.path.basename(f)[:28]}... trained on raw returns by job {job}",
              "_EWMA_" in os.path.basename(f) and fresh, f"written {pd.Timestamp.fromtimestamp(os.path.getmtime(f)):%H:%M:%S}, job start {start:%H:%M:%S}")
    logs = glob.glob(f"{ROOT}/logs/v2_rare_{key}_{job}.out")
    txt = open(logs[0]).read() if logs else ""
    saved_at, loaded_at = txt.find("Model saved to ckpt_new/score_"), txt.find("Model loaded from ckpt_new/score_")
    check(f"inclusion/rare {key}: base model (M0) trained fresh in this job before any load",
          "STEP 2: Diffusion Model Training" in txt and saved_at >= 0 and (loaded_at < 0 or saved_at < loaded_at))
    main_src = open(C + "/main.py").read()
    check(f"inclusion/rare {key}: base model (M0) trained unconditionally in main.py",
          "diffusion_model.train(\n            train_data=train_data,\n            batch_size" in main_src)
    for ch in ["macro_change", "ewma_mean_change", "ewma_vol_change", "pca_change"]:
        new = pd.read_csv(f"{C}/explore/standardization_results/data_for_diffusion_{ch}.csv", parse_dates=["Date"]).set_index("Date")
        old = pd.read_csv(f"{ROOT}/explore/standardization_results/data_for_diffusion_{ch}.csv", parse_dates=["Date"]).set_index("Date")
        raw_ok = np.allclose(new[TICKERS].to_numpy(), RAW.reindex(new.index)[TICKERS].to_numpy())
        col0 = np.allclose(new.iloc[:, 0], old.iloc[:, 0], equal_nan=True)
        check(f"inclusion/rare {key}: {ch} assets = raw log returns, event column unchanged", raw_ok and col0)


def standardization():
    metas = {m: json.load(open(f"{STD}/{d}/meta.json")) for m, d in METHODS.items()}
    check("embedding/std: identical NN hyperparameters for S0-S4", len({json.dumps(x["hyperparams"], sort_keys=True) for x in metas.values()}) == 1,
          json.dumps(metas["s0"]["hyperparams"]))
    check("embedding/std: identical fit dates for S0-S4", len({(x["n_fit_dates"], x["fit_start"], x["fit_end"]) for x in metas.values()}) == 1,
          f"{metas['s0']['n_fit_dates']} dates {metas['s0']['fit_start']}..{metas['s0']['fit_end']}")
    from nn_standardize import MLP, NLL_Stationarity
    for m, d in METHODS.items():
        x = metas[m]
        check(f"embedding/std: {m} fit on its OWN state", x["macro_series"].endswith(f"{d}/daily_state.csv"), x["macro_series"].split("/")[-2])
        sd = torch.load(f"{STD}/{d}/standardizer.pt", weights_only=False)
        st = pd.read_csv(x["macro_series"], index_col=0, parse_dates=True)["daily_state"]
        mu = pd.read_csv(f"{STD}/{d}/mu.csv", parse_dates=["Date"]).set_index("Date")
        sg = pd.read_csv(f"{STD}/{d}/sigma.csv", parse_dates=["Date"]).set_index("Date")
        z = pd.read_csv(f"{STD}/{d}/residuals.csv", parse_dates=["Date"]).set_index("Date")
        dates = mu.dropna().index[::97]
        v = torch.tensor(((st.reindex(dates).to_numpy() - sd["m_mean"]) / sd["m_std"]).reshape(-1, 1), dtype=torch.float32)
        worst = 0.0
        for t in TICKERS:
            hp = sd["hyperparams"]
            model = NLL_Stationarity(MLP(1, hp["hidden"], hp["n_layers"]), MLP(1, hp["hidden"], hp["n_layers"]), delta=hp["delta"])
            model.mu_net.load_state_dict(sd["nets"][t]["mu_net"]); model.sigma_net.load_state_dict(sd["nets"][t]["sigma_net"])
            with torch.no_grad():
                a, b = model.joint_forward(v)
            worst = max(worst, float(np.max(np.abs(a.numpy() - mu.loc[dates, t].to_numpy()))),
                        float(np.max(np.abs(b.numpy() - sg.loc[dates, t].to_numpy()))))
        inv = float(np.nanmax(np.abs((mu + sg * z - RAW.reindex(z.index)[TICKERS]).to_numpy())))
        check(f"embedding/std: {m} saved networks reproduce mu/sigma; r = mu + sigma*z", worst < 1e-5 and inv < 1e-5,
              f"net err {worst:.1e}, inverse err {inv:.1e}, z std {np.nanstd(z.to_numpy()):.3f}")


def embedding_rare():
    for key in ["emb_nc", "emb_c"]:
        C = COPIES[key]
        for m in METHODS:
            a = f"{C}/explore/standardization_results/data_for_diffusion_exp2_{m}_ownstd.csv"
            b = f"{ROOT}/explore/standardization_results/data_for_diffusion_exp2_{m}_ownstd.csv"
            same = os.path.exists(a) and subprocess.run(["cmp", "-s", a, b]).returncode == 0
            check(f"embedding/rare {key}: {m} own-residual CSV in copy (verified build)", same)
        check(f"embedding/rare {key}: fresh copy, no base checkpoint to reuse", not os.listdir(C + "/ckpt_new"))
        check(f"embedding/rare {key}: no stale config/CSV backups in copy",
              not glob.glob(C + "/config/*orig_backup*") and not glob.glob(C + "/explore/standardization_results/*orig_backup*"))


def embedding_conditional_stub():
    """Run diffusion_adapters.main() for all 5 methods with ONLY DiffusionModel.train stubbed; check that
    each method's training data are exactly its own residual windows and its conditioner is causal."""
    import diffusion_adapters as da
    import common
    from models.diffusion_model import DiffusionModel
    seen = {}
    def stub(self, train_data, condition_data=None, end_dates=None, loss_csv_path=None, **kw):
        seen[len(seen)] = (train_data.numpy(), condition_data.numpy(), pd.DatetimeIndex(end_dates))
        os.makedirs(os.path.dirname(loss_csv_path), exist_ok=True)
    DiffusionModel.train = stub
    out = "/tmp/user/25913/claude-25913/-scr-kadenwu-Conditional-diffusion/675cecbe-6913-4b68-8281-dd675dbd72b6/scratchpad/preflight_cond_emb"
    orig = common.get_default_config
    def cpu_cfg():
        c = orig(); c.diffusion.device = "cpu"; return c
    common.get_default_config = cpu_cfg
    sys.argv = ["diffusion_adapters.py", "--experiment", "full", "--mode", "train", "--out-dir", out]
    da.main()
    cal = pd.DatetimeIndex(RAW.index)
    resid = {m: pd.read_csv(f"{STD}/{d}/residuals.csv", parse_dates=["Date"]).set_index("Date").reindex(cal)[TICKERS].to_numpy()
             for m, d in METHODS.items()}
    states = {m: pd.read_csv(f"{FULL}/extras/exp2_results/{d}/daily_state.csv", index_col=0, parse_dates=True)["daily_state"]
              .reindex(cal).to_numpy() for m, d in METHODS.items()}
    for i, m in enumerate(METHODS):
        X, c, ends = seen[i]
        st = cal.get_indexer(ends) - 9
        Z = np.stack([resid[m][s:s + 10].T for s in st])
        scale = X[0, 0, 0] / Z[0, 0, 0]
        own = np.allclose(X / scale, Z, rtol=1e-4, atol=1e-6)
        not_s0 = m == "s0" or not np.allclose(X / scale, np.stack([resid["s0"][s:s + 10].T for s in st]), atol=1e-3)
        not_raw = not np.allclose(X / scale, np.stack([RAW[TICKERS].to_numpy()[s:s + 10].T for s in st]), atol=1e-3)
        cw = np.stack([states[m][s - 11:s - 1] for s in st])   # 10 values ending start - gap - 1 (gap 1)
        causal = np.allclose(c * cw.std() + cw.mean(), cw, rtol=1e-4, atol=1e-6 * np.abs(cw).max())
        check(f"embedding/conditional-only: {m} trains on its OWN residual windows (not S0's, not raw)",
              own and not_s0 and not_raw, f"{len(X)} train windows")
        check(f"embedding/conditional-only: {m} conditioner = its state, 10 values ending 2 rows before window", causal)
    check("embedding/conditional-only: identical train windows across S0-S4", len({tuple(seen[i][2]) for i in seen}) == 1)


def slurm_and_dirs():
    for f in ["run_v2_rare_inc_nc.slurm", "run_v2_rare_inc_c.slurm", "run_v2_rare_emb_nc.slurm", "run_v2_rare_emb_c.slurm",
              "run_v2_cond_full_emb.slurm"]:
        p = f"{ROOT}/{f}"
        ok = os.path.exists(p)
        if ok:
            txt = open(p).read()
            cds = [l.split("cd ", 1)[1].strip().strip('"') for l in txt.splitlines() if l.startswith("cd ")]
            ok = all(os.path.isdir(c) for c in cds) and "--nodelist=soal-5" in txt
        check(f"slurm: {f} exists, cd targets exist", ok)
    for d in [FULL + "/causal/macro_embed/conditional_only", FULL + "/causal/macro_embed/rare_event",
              FULL + "/noncausal/macro_embed/rare_event", FULL + "/causal/macro_inclusion/rare_event",
              FULL + "/noncausal/macro_inclusion/rare_event"]:
        check(f"output dir ready and holds no stale results: {d.replace(FULL + '/', '')}", os.path.isdir(d) and not os.listdir(d))


if __name__ == "__main__":
    for fn in [inclusion_conditional, lambda: inclusion_rare("inc_nc", False), lambda: inclusion_rare("inc_c", True),
               standardization, embedding_rare, embedding_conditional_stub, slurm_and_dirs]:
        try:
            fn()
        except Exception as ex:  # a crash is a failure, never a skip
            check(f"{getattr(fn, '__name__', 'check')} crashed", False, repr(ex)[:300])
    n_fail = sum(not ok for _, ok, _ in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    sys.exit(1 if n_fail else 0)
