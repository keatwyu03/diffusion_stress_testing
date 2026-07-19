import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def monthly_first_pc(macro):
    """First PC of the standardized monthly macro panel (per group — growth and
    inflation each get their OWN PCA, never a joint one), sign convention:
    loadings positive on average. Shared starting point of BOTH latent-state
    methods, so tracking_regression and state_space provably use identical
    monthly factors."""
    macro = macro.dropna()
    X_scaled = StandardScaler().fit_transform(macro)
    pca = PCA(n_components=1)
    factor = pd.Series(pca.fit_transform(X_scaled)[:, 0], index=macro.index, name="zm")
    loadings = pd.Series(pca.components_[0], index=macro.columns)
    if loadings.mean() < 0:
        factor, loadings = -factor, -loadings
    return factor, loadings, pca.explained_variance_ratio_[0]


class TrackingRegression:
    def __init__(self, macro, daily):
        self.macro = macro.dropna()
        self.daily = daily

    def fit(self):
        # monthly factor: first PC of the standardized macro panel
        self.factor, self.loadings, self.explained_var = monthly_first_pc(self.macro)

        # tracking regression: z_{m+1} on z_m and monthly-summed daily returns
        monthly = self.daily.resample("MS").sum()

        z = pd.concat([self.factor, monthly], axis=1)
        z["m+1"] = z["zm"].shift(-1)
        z = z.dropna()

        X = sm.add_constant(z.drop(columns="m+1"))
        self.model = sm.OLS(z["m+1"], X).fit()
        self.betas = self.model.params[self.daily.columns]

        # daily tracking portfolio returns
        self.ut = self.daily @ self.betas
        return self.ut
