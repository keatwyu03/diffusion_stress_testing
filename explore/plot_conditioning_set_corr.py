"""
Causal event-window correlation figures for the cached conditioning sets -- NO refit: reads
explore/conditioning_sets/<set>/macro_data_new.csv (built once by build_conditioning_sets.py).

Saves explore/conditioning_sets/<set>/real_corr_event_vs_all_<top>_lag<gap>.png and a matching .json with the
block-average correlations. event_causal is always True; only the event threshold and event_lag_gap vary.

Usage:  python explore/plot_conditioning_set_corr.py --lag 0 [--sets full,cpi_only,inflation_only] [--tops 0.075,0.05]
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
from config import get_default_config  # noqa: E402
import build_conditioning_sets as B  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lag", type=int, required=True)
    ap.add_argument("--sets", default=",".join(B.SETS))
    ap.add_argument("--tops", default="0.075,0.05")
    a = ap.parse_args()
    for name in a.sets.split(","):
        d = os.path.join(B.OUT_ROOT, name)
        if not os.path.exists(os.path.join(d, "macro_data_new.csv")):
            raise SystemExit(f"{d} not built -- run build_conditioning_sets.py --set {name} first")
        for top in [float(x) for x in a.tops.split(",")]:
            cfg = get_default_config()
            for k, v in B.SETS[name].items():
                setattr(cfg.data, k, v)
            cfg.data.event_causal = True
            cfg.data.event_lag_gap = a.lag
            tag = f"top{top * 100:g}pct_lag{a.lag}"
            res = B.plot_corr(cfg, os.path.join(d, "macro_data_new.csv"), os.path.join(d, f"real_corr_event_vs_all_{tag}.png"), top)
            json.dump({"set": name, "event_causal": True, "event_lag_gap": a.lag, "event_top": top, "panels": res},
                      open(os.path.join(d, f"real_corr_event_vs_all_{tag}.json"), "w"), indent=2)
            print(f"RESULT {name:15s} {tag:16s} " + " | ".join(
                f"{k.replace('Real ', '')}: n={v['n']} overall {v['overall']:.3f} (tech {v['tech']:.3f}, staples {v['staples']:.3f}, cross {v['cross']:.3f})"
                for k, v in res.items()), flush=True)
