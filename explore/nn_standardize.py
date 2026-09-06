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
    
    def fit(self, r, m, n_epochs=1000, batch_size=128, block_size=35, min_delta=0.01,
            verbose = True, desc = "fit"):
        """Trains up to n_epochs (a cap, not a target). Every block_size epochs,
        compares that block's average loss to the previous block's; two
        consecutive blocks that each fail to improve by at least min_delta stop
        training early. The single best (global-minimum-loss) checkpoint seen
        is restored before returning, since the run may continue past the best
        epoch before the stop triggers. Same convergence pattern as
        HFunctionDirectTrainer.fit() (models/hfunction_direct.py)."""
        n = len(r)

        best_loss = float("inf")
        best_state = None
        block_losses = []
        prev_block_avg = None
        stagnant_blocks = 0

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

            if loss < best_loss:
                best_loss = loss
                best_state = {
                    "mu_net": {k: v.detach().clone() for k, v in self.mu_net.state_dict().items()},
                    "sigma_net": {k: v.detach().clone() for k, v in self.sigma_net.state_dict().items()},
                }

            block_losses.append(loss)
            if len(block_losses) == block_size:
                block_avg = sum(block_losses) / block_size
                block_losses = []

                if prev_block_avg is not None:
                    improvement = prev_block_avg - block_avg
                    if improvement < min_delta:
                        stagnant_blocks += 1
                        if stagnant_blocks >= 2:
                            break
                    else:
                        stagnant_blocks = 0

                prev_block_avg = block_avg

        if best_state is not None:
            self.mu_net.load_state_dict(best_state["mu_net"])
            self.sigma_net.load_state_dict(best_state["sigma_net"])
            self.best_loss = best_loss

    @torch.no_grad()
    def standardize(self, r, m):
        mu, sigma = self.joint_forward(m)
        return (r - mu) / sigma


# ─────────────────────────────────────────────────────────────────────────────
# Run on the conditioning series + ticker returns in explore/macro_data_new.csv
# ─────────────────────────────────────────────────────────────────────────────

def load_data(csv_path=None, cond_col="m_t"):
    """(dates, m, returns_by_ticker) from the CSV, NaN conditioning rows dropped.

    Reads macro_data_new.csv (import_data.py's output): `m_t` is the latent
    macro state, every other numeric column is that ticker's daily LOG-RETURN
    (import_data.py writes np.log(P/P.shift(1))). This step standardizes those
    returns against m_t: z = (r - mu(m_t)) / sigma(m_t).
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


def run_ticker(r, m, ticker, hidden=64, n_layers=2, n_epochs=1000, batch_size=128,
               delta=1e-3, lr=1e-3, seed=0, verbose=True):
    torch.manual_seed(seed)
    p = m.shape[1]

    m_full = (m - m.mean(0)) / m.std(0)
    model = NLL_Stationarity(MLP(p, hidden, n_layers), MLP(p, hidden, n_layers),
                             delta=delta, lr=lr)
    model.fit(r, m_full, n_epochs=n_epochs, batch_size=batch_size,
             verbose=verbose, desc=f"fitting {ticker}")
    z = model.standardize(r, m_full)
    with torch.no_grad():
        mu, sigma = model.joint_forward(m_full)

    return model, z, mu, sigma


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
        # same file residual_main.py writes, so both entry points agree and
        # config.data.csv_path (which the diffusion reads) has one target
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "standardization_results", "nn_standardized_macro.csv")

    out = pd.DataFrame({t: z.detach().cpu().numpy() for t, z in residuals_by_ticker.items()})
    out.insert(0, "Date", dates.reset_index(drop=True))
    if m is not None:
        out.insert(1, "m_t", m.detach().cpu().numpy().flatten())
    out.to_csv(csv_path, index=False)
    print(f"wrote {len(out)} rows x {len(residuals_by_ticker)} tickers to {csv_path}")
    return out


def save_diffusion_csv(out, csv_path=None):
    """Write the final conditional-diffusion input CSV (config.data.csv_path):
    col 0 = Date, col 1 = m_t latent macro state, remaining cols = the
    standardized asset columns. This is the file main.py / DataProcessor read.
    Overwrites: it's meant to be regenerated whenever you flip
    config.data.latent_standardized between experiment arms. The checkpoint
    naming ("_LS" suffix via get_run_tag) is what keeps the two arms' trained
    models from colliding, not the CSV name."""
    import os

    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "standardization_results", "data_for_diffusion.csv")

    out.to_csv(csv_path, index=False)
    print(f"wrote conditional-diffusion input ({len(out)} rows, "
          f"{out.shape[1] - 1} cols after Date) to {csv_path}")
    return csv_path


def plain_zscore(returns_by_ticker):
    """Unconditional per-ticker z-score of the raw log-returns:
    z = (r - mean(r)) / std(r), whole-series stats. Used when
    config.data.latent_standardized is False -- m_t is still computed and
    kept as column 0 of the output CSV, it just isn't used to standardize."""
    out = {}
    for t, r in returns_by_ticker.items():
        out[t] = (r - r.mean()) / r.std()
    return out


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import get_default_config

    _cfg = get_default_config()
    latent_std = _cfg.data.latent_standardized

    dates, m, returns = load_data()
    print(f"{len(m)} days, {dates.iloc[0].date()} to {dates.iloc[-1].date()}, "
          f"{len(returns)} tickers")
    print(f"latent_standardized = {latent_std}  "
          f"({'NN macro-conditional residuals' if latent_std else 'plain unconditional z-score'})\n")

    if latent_std:
        z_by_ticker = {}
        for ticker, r in returns.items():
            _, z, _, _ = run_ticker(r, m, ticker)
            z_by_ticker[ticker] = z
        # keep writing the NN-residual diagnostics file in this arm
        out = save_standardized_residuals(z_by_ticker, dates, m=m)
    else:
        z_by_ticker = plain_zscore(returns)
        out = pd.DataFrame({t: z.detach().cpu().numpy() for t, z in z_by_ticker.items()})
        out.insert(0, "Date", dates.reset_index(drop=True))
        out.insert(1, "m_t", m.detach().cpu().numpy().flatten())

    save_diffusion_csv(out)
