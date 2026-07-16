import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class TrackingRegression:
    def __init__(self, macro, daily):
        self.macro = macro.dropna()
        self.daily = daily

    def fit(self):
        # monthly factor: first PC of the standardized macro panel
        X_scaled = StandardScaler().fit_transform(self.macro)
        pca = PCA(n_components=1)
        factors = pca.fit_transform(X_scaled)

        self.factor = pd.Series(factors[:, 0], index=self.macro.index, name="zm")
        self.loadings = pd.Series(pca.components_[0], index=self.macro.columns)
        self.explained_var = pca.explained_variance_ratio_[0]

        # sign convention: positive loadings on average
        if self.loadings.mean() < 0:
            self.factor *= -1
            self.loadings *= -1

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
