import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


class MLP(nn.Module): 
    def __init__(self, p, hidden, n_layers, out = 1):
        super().__init__()
        
        layers = []
        d = p
        for i in range(n_layers):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.SiLU())
            d = hidden
        layers.append(nn.Linear(d, out))

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)



class NLL_Stationarity:
    def __init__(self, mu_net, sigma_net, delta = 1e-3, lr = 1e-3, sigma_wd = 1e-4, mu_wd = 1e-4):
        self.mu_net = mu_net
        self.sigma_net = sigma_net
        self.delta = delta
        self.lr = lr
        self.sigma_wd = sigma_wd
        self.mu_wd = mu_wd

        self.optimizer = optim.Adam([
            {'params' : self.mu_net.parameters(), 'weight_decay' : self.mu_wd},
            {'params' : self.sigma_net.parameters(), 'weight_decay' : self.sigma_wd}
        ], lr = self.lr)
    
    def joint_forward(self, m): 
        mu = self.mu_net(m)
        sigma = self.delta + F.softplus(self.sigma_net(m))
        return mu, sigma
    
    def nll(self, r, m):
        mu, sigma = self.joint_forward(m)
        return ((r - mu) ** 2 / (2 * sigma ** 2) + torch.log(sigma)).mean()

    def step(self, r, m):
        self.optimizer.zero_grad()
        loss = self.nll(r, m)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def fit(self, r, m, n_epochs, batch_size, verbose = True, desc = "fit"):
        n = len(r)
        epochs = tqdm(range(n_epochs), desc=desc, leave=False) if verbose else range(n_epochs)
        for epoch in epochs:
            perm = torch.randperm(n)
            total = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i : i+batch_size]
                total += self.step(r[idx], m[idx]) * len(idx)
            loss = total / n
            if verbose:
                epochs.set_postfix(nll=f"{loss:.4f}")
        
    @torch.no_grad()
    def standardize(self, r, m):
        mu, sigma = self.joint_forward(m)
        return (r - mu) / sigma

    @torch.no_grad()
    def macro_bins(self, z, m, n_bins = 10):
        z = z.detach().flatten().double()
        m = m.detach().double()
        if m.ndim > 1:
            if m.shape[1] != 1:
                raise ValueError(
                    f"macro binning needs a scalar conditioning series, got m with "
                    f"{m.shape[1]} columns — bin on one column or a projection of them"
                )
            m = m.flatten()
        if len(m) != len(z):
            raise ValueError(f"z/m length mismatch: {len(z)} vs {len(m)}")

        qs = torch.linspace(0, 1, n_bins + 1, dtype=torch.float64)[1:-1]
        edges = torch.quantile(m, qs)
        idx = torch.bucketize(m, edges)

        counts, bin_means, bin_vars, bin_lo, bin_hi = [], [], [], [], []
        for b in range(n_bins):
            sel = z[idx == b]
            counts.append(len(sel))
            bin_means.append(sel.mean() if len(sel) else torch.tensor(float("nan"), dtype=torch.float64))
            bin_vars.append(sel.var(unbiased=False) if len(sel) > 1 else torch.tensor(float("nan"), dtype=torch.float64))
            mb = m[idx == b]
            bin_lo.append(float(mb.min()) if len(mb) else float("nan"))
            bin_hi.append(float(mb.max()) if len(mb) else float("nan"))

        bin_means = torch.stack(bin_means)
        bin_vars = torch.stack(bin_vars)

        ok_m = ~torch.isnan(bin_means)
        ok_v = ~torch.isnan(bin_vars)
        mu_bar = bin_means[ok_m].mean()
        s_mu = ((bin_means[ok_m] - mu_bar) ** 2).mean().sqrt()
        v_bar = bin_vars[ok_v].mean()
        s_v = ((bin_vars[ok_v] - v_bar) ** 2).mean().sqrt()

        return {
            "n": len(z),
            "n_bins": n_bins,
            "counts": counts,
            "bin_lo": bin_lo,
            "bin_hi": bin_hi,
            "bin_means": bin_means,
            "bin_vars": bin_vars,
            "mu_bar": float(mu_bar),
            "s_mu": float(s_mu),
            "v_bar": float(v_bar),
            "s_v": float(s_v),
        }

    @staticmethod
    def print_macro_bins(d, label = "z"):
        print(f"Macro-relative stationarity — {label}")
        print(f"  {d['n']} points in {d['n_bins']} quantile bins of m")

        print("\n  per macro bin")
        print("      bin      n        m range          E[z|G]   Var[z|G]")
        for b in range(d["n_bins"]):
            rng = f"[{d['bin_lo'][b]:+.2f}, {d['bin_hi'][b]:+.2f}]"
            print(f"    {b:>5}{d['counts'][b]:>7}   {rng:>18}"
                  f"{float(d['bin_means'][b]):>+10.4f}{float(d['bin_vars'][b]):>11.4f}")

        print("\n  conditional mean  E[z|G]     (target: 0 in every bin)")
        print(f"    average across bins     {d['mu_bar']:+.4f}")
        print(f"    sd across bins          {d['s_mu']:.4f}")

        print("\n  conditional variance Var[z|G]  (target: 1 in every bin)")
        print(f"    average across bins     {d['v_bar']:.4f}")
        print(f"    sd across bins          {d['s_v']:.4f}")

    @torch.no_grad()
    def time_blocks(self, z, max_lag = 10, n_blocks = 10):
        z = z.detach().flatten().double()
        n_blocks = min(n_blocks, len(z) // (max_lag + 2))
        edges = torch.linspace(0, len(z), n_blocks + 1).long()

        means, vars_ = [], []
        rows = []
        for b in range(n_blocks):
            x = z[edges[b] : edges[b + 1]]
            means.append(x.mean())
            vars_.append(x.var(unbiased=False))
            c = x - x.mean()          # each block is centred on its OWN mean
            nb = len(c)
            rows.append(torch.stack([
                (c[k:] * c[: nb - k]).mean() for k in range(1, max_lag + 1)
            ]))
        acov = torch.stack(rows)                                    # (B, max_lag)
        block_means = torch.stack(means)                            # (B,)
        block_vars = torch.stack(vars_)                             # (B,)

        gamma_bar = acov.mean(dim=0)                                # per-lag average
        s_gamma = ((acov - gamma_bar) ** 2).mean(dim=0).sqrt()      # per-lag sd

        return {
            "n": len(z),
            "max_lag": max_lag,
            "n_blocks": n_blocks,
            "block_size": len(z) / n_blocks,
            "block_means": block_means,
            "block_vars": block_vars,
            "acov": acov,
            "gamma_bar": gamma_bar,
            "s_gamma": s_gamma,
        }

    @staticmethod
    def print_time_blocks(d, label = "z"):
        acov, K = d["acov"], d["max_lag"]
        print(f"Time-block stationarity — {label}")
        print(f"  {d['n']} points in {d['n_blocks']} time blocks of ~{d['block_size']:.0f}")

        print("\n  per time block")
        print("    block          mean      variance")
        for b in range(d["n_blocks"]):
            print(f"    {b:>5}{float(d['block_means'][b]):>+14.6f}{float(d['block_vars'][b]):>14.6f}")

        print("\n  gamma_b(k) — rows are time blocks, columns are lags")
        print("    block" + "".join(f"{k:>9}" for k in range(1, K + 1)))
        for b in range(d["n_blocks"]):
            print(f"    {b:>5}" + "".join(f"{float(v):>9.4f}" for v in acov[b]))

        print("\n    " + "-" * (5 + 9 * K))
        print("   avg  " + "".join(f"{float(v):>9.4f}" for v in d["gamma_bar"]))
        print("    sd  " + "".join(f"{float(v):>9.4f}" for v in d["s_gamma"]))
        print("\n  gamma(k) should not depend on t: read DOWN a column (one lag,")
        print("  every block) for drift; the sd row is how much that lag moves.")


def print_before_after(r, z, label="z"):
    """Unconditional mean/variance of the raw return r vs the standardized
    residual z (no macro binning — just the plain moments over all points)."""
    r = r.detach().flatten().double()
    z = z.detach().flatten().double()

    r_mean, r_var = r.mean().item(), r.var(unbiased=False).item()
    z_mean, z_var = z.mean().item(), z.var(unbiased=False).item()

    print(f"Before/after standardization — {label}")
    print(f"{'':<12}{'raw r (before)':>18}{'standardized z (after)':>26}")
    print(f"  {'mean':<10}{r_mean:>18.4g}{z_mean:>26.4g}")
    print(f"  {'variance':<10}{r_var:>18.4g}{z_var:>26.4g}")


# ─────────────────────────────────────────────────────────────────────────────
# Run on the conditioning series + ticker returns in explore/macro_data_new.csv
# ─────────────────────────────────────────────────────────────────────────────

def load_data(csv_path=None, cond_col="m_t"):
    """(dates, m, returns_by_ticker) from the CSV, NaN conditioning rows dropped.

    `m_t` is the latent macro state written by import_data.py. Every other
    numeric column is a ticker's daily returns.
    """
    import os
    import pandas as pd

    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "macro_data_new.csv")

    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df[df[cond_col].notna()].reset_index(drop=True)

    tickers = [c for c in df.columns if c not in ("Date", cond_col)]
    m = torch.tensor(df[[cond_col]].to_numpy(), dtype=torch.float32)   # (n, 1)
    returns = {t: torch.tensor(df[t].to_numpy(), dtype=torch.float32) for t in tickers}
    return df["Date"], m, returns


def run_ticker(r, m, ticker, train_frac=0.7, hidden=64, n_layers=2,
               n_epochs=500, batch_size=128, delta=1e-3, lr=1e-3,
               n_bins=10, max_lag=10, n_time_blocks=10, seed=0,
               print_split=False, verbose=True):

    torch.manual_seed(seed)
    p = m.shape[1]

    # ---- FULL-SERIES MODEL: the one the results rest on -------------------
    # This is in-sample characterisation. The question is whether a
    # location-scale decomposition in m CAN absorb the macro-driven
    # nonstationarity, not whether it forecasts, so every observation is
    # treated identically and both diagnostics run off this single fit.
    m_full = (m - m.mean(0)) / m.std(0)
    model_full = NLL_Stationarity(MLP(p, hidden, n_layers), MLP(p, hidden, n_layers),
                                  delta=delta, lr=lr)
    model_full.fit(r, m_full, n_epochs=n_epochs, batch_size=batch_size,
                   verbose=verbose, desc=f"fitting {ticker} (full)")
    z_full = model_full.standardize(r, m_full)

    d_tb = model_full.time_blocks(z_full, max_lag=max_lag, n_blocks=n_time_blocks)

    # ---- BASELINE: same time blocks, on the RAW (unstandardized) return ----
    # Without this, "how much nonstationarity did standardization remove" has
    # no "before" to compare "after" against.
    d_tb_raw = model_full.time_blocks(r, max_lag=max_lag, n_blocks=n_time_blocks)

    if verbose:
        model_full.print_time_blocks(d_tb_raw, label=f"{ticker} — RAW (before standardization)")
        print()
        model_full.print_time_blocks(d_tb, label=f"{ticker} — FULL SERIES (after standardization)")
        print()
        print_before_after(r, z_full, label=ticker)

    # ---- TRAIN/TEST SPLIT: secondary, printed only on request -------------
    d_tb_in = d_tb_out = None
    if print_split:
        cut = int(train_frac * len(r))
        m_scaled = (m - m[:cut].mean(0)) / m[:cut].std(0)
        model = NLL_Stationarity(MLP(p, hidden, n_layers), MLP(p, hidden, n_layers),
                                 delta=delta, lr=lr)
        model.fit(r[:cut], m_scaled[:cut], n_epochs=n_epochs, batch_size=batch_size,
                  verbose=verbose, desc=f"fitting {ticker} (split)")

        print()
        z_in = model.standardize(r[:cut], m_scaled[:cut])
        d_tb_in = model.time_blocks(z_in, max_lag=max_lag, n_blocks=n_time_blocks)
        model.print_time_blocks(d_tb_in, label=f"{ticker} — IN-SAMPLE (train)")

        print()
        z_out = model.standardize(r[cut:], m_scaled[cut:])
        d_tb_out = model.time_blocks(z_out, max_lag=max_lag, n_blocks=n_time_blocks)
        model.print_time_blocks(d_tb_out, label=f"{ticker} — HELD-OUT (test)")

    return model_full, z_full, (d_tb, d_tb_in, d_tb_out, d_tb_raw)


def save_standardized_residuals(residuals_by_ticker, dates, m=None, csv_path=None):
    """Write full-series standardized residuals z_full to a CSV (Date, m_t,
    then one column per ticker), so downstream consumers (e.g.
    bfk_stationarity.py) don't need to refit the models themselves. `m` is
    the raw conditioning series (same m fed to every ticker's mu_net/sigma_net
    in run_ticker) — included so the macro state each z was standardized
    against is visible alongside the residuals. Always overwrites csv_path.
    """
    import os

    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "nn_standardized_macro.csv")

    out = pd.DataFrame({t: z.detach().cpu().numpy() for t, z in residuals_by_ticker.items()})
    out.insert(0, "Date", dates.reset_index(drop=True))
    if m is not None:
        out.insert(1, "m_t", m.detach().cpu().numpy().flatten())
    out.to_csv(csv_path, index=False)
    print(f"wrote {len(out)} rows x {len(residuals_by_ticker)} tickers to {csv_path}")
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--print-split-results", action="store_true",
                        help="also fit a 70/30 split model and print its in-sample "
                             "and held-out macro bins")
    args = parser.parse_args()

    dates, m, returns = load_data()
    print(f"{len(m)} days, {dates.iloc[0].date()} to {dates.iloc[-1].date()}, "
          f"{len(returns)} tickers\n")

    results = {}
    z_by_ticker = {}
    for ticker, r in returns.items():
        print("=" * 72)
        _, z_full, results[ticker] = run_ticker(r, m, ticker,
                                                 print_split=args.print_split_results)
        z_by_ticker[ticker] = z_full
        print()

    save_standardized_residuals(z_by_ticker, dates, m=m)
    print()

    def _block_spread(d, key):
        """(level, sd-across-blocks) for block_means or block_vars."""
        vals = d[key]
        level = vals.mean().item()
        sd = ((vals - vals.mean()) ** 2).mean().sqrt().item()
        return level, sd

    def summary_table(title, which):
        print("=" * 72)
        print(f"SUMMARY — {title}")
        print("=" * 72)
        print("           |----- mean across blocks -----|--- variance across blocks ---|")
        print(f"  {'ticker':<8}{'avg':>12}{'sd':>13}{'avg':>13}{'sd':>13}")
        for ticker, res in results.items():
            d = res[which]
            mu_bar, s_mu = _block_spread(d, "block_means")
            v_bar, s_v = _block_spread(d, "block_vars")
            print(f"  {ticker:<8}{mu_bar:>+12.4f}{s_mu:>13.4f}"
                  f"{v_bar:>13.4f}{s_v:>13.4f}")
        print()

    summary_table("FULL SERIES standardized innovations z", 0)
    print("  target: mean ~0 and variance ~1 in EVERY time block.")
    print("  the sd columns are the test — they measure how much the moments")
    print("  still drift over time, i.e. what standardization missed.")

    print("=" * 72)
    print("SUMMARY — before/after standardization (variance-flatness improvement)")
    print("=" * 72)
    print("  CV = block-variance sd / block-variance level — lower is flatter over time.")
    print(f"  {'ticker':<8}{'CV before':>14}{'CV after':>14}{'improvement':>16}")
    for ticker, res in results.items():
        d_z_t, d_raw_t = res[0], res[3]
        v_bar_raw, s_v_raw = _block_spread(d_raw_t, "block_vars")
        v_bar_z, s_v_z = _block_spread(d_z_t, "block_vars")
        cv_raw = s_v_raw / abs(v_bar_raw) if v_bar_raw else float("nan")
        cv_z = s_v_z / abs(v_bar_z) if v_bar_z else float("nan")
        improvement = f"{cv_raw / cv_z:.2f}x" if cv_z > 0 else "n/a"
        print(f"  {ticker:<8}{cv_raw:>14.4g}{cv_z:>14.4g}{improvement:>16}")
    print("  improvement < 1x means standardization made variance-flatness WORSE")
    print("  for that ticker — worth checking individually if it shows up.\n")

    if args.print_split_results:
        print()
        summary_table("IN-SAMPLE (train) standardized innovations z", 1)
        summary_table("HELD-OUT (test) standardized innovations z", 2)
