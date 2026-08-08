import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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
    
    def fit(self, r, m, n_epochs, batch_size, verbose = True):
        n = len(r)
        for epoch in range(n_epochs):
            perm = torch.randperm(n)
            total = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i : i+batch_size]
                total += self.step(r[idx], m[idx]) * len(idx)
            loss = total / n
            if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
                print(f"Epoch {epoch} | Loss: {loss:.4f}")
        
    @torch.no_grad()
    def standardize(self, r, m):
        mu, sigma = self.joint_forward(m)
        return (r - mu) / sigma

    @torch.no_grad()
    def diagnostics(self, z, max_lag = 10, n_blocks = None):
        z = z.detach().flatten().double()

        L = max_lag + 1
        n_eff = len(z) - max_lag
        stack = torch.stack([z[max_lag - j : max_lag - j + n_eff] for j in range(L)])
        centered = stack - stack.mean(dim=1, keepdim=True)
        cov_matrix = (centered @ centered.T) / n_eff

        if n_blocks is None:
            n_blocks = max(2, int(len(z) ** 0.5))
        n_blocks = min(n_blocks, len(z) // 2)
        edges = torch.linspace(0, len(z), n_blocks + 1).long()
        blocks = [z[edges[i] : edges[i + 1]] for i in range(n_blocks)]
        block_means = torch.tensor([b.mean() for b in blocks], dtype=torch.float64)
        block_vars = torch.tensor([b.var(unbiased=False) for b in blocks], dtype=torch.float64)

        return {
            "n": len(z),
            "max_lag": max_lag,
            "mean": float(z.mean()),
            "var": float(z.var(unbiased=False)),
            "cov_matrix": cov_matrix,
            "n_blocks": n_blocks,
            "block_size": len(z) / n_blocks,
            "block_means": block_means,
            "block_vars": block_vars,
        }

    @staticmethod
    def print_diagnostics(d, label = "z"):
        cov = d["cov_matrix"]
        L = cov.shape[0]

        print(f"Stationarity diagnostics — {label}  (n = {d['n']})")
        print(f"  mean {d['mean']:+.4f}   target 0")
        print(f"  var  {d['var']:.4f}   target 1")

        bm, bv = d["block_means"], d["block_vars"]
        print(f"\n  {d['n_blocks']} blocks of ~{d['block_size']:.0f} points")
        print("    block      mean       var")
        for i, (mu, v) in enumerate(zip(bm, bv)):
            print(f"    {i:>5}   {float(mu):+8.4f}  {float(v):8.4f}")

        print(f"\n  Cov(z_t-a, z_t-b) — Toeplitz (constant diagonals) under stationarity")
        header = "       " + "".join(f"{b:>8}" for b in range(L))
        print(header)
        for a in range(L):
            row = "".join(f"{float(cov[a, b]):>8.3f}" for b in range(L))
            print(f"  a={a:<3}{row}")
