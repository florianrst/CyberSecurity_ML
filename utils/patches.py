"""
Runtime patches for third-party library bugs.
Call apply_patches() before using the affected libraries.
"""


def _fix_prince_famd():
    """
    Fix prince.FAMD.fit: sparse arithmetic changes the fill_value to nan,
    which floods the dense matrix with NaN when to_dense() is called.
    row_coordinates() already uses .fillna(0) — fit() was missing it.
    """
    import numpy as np
    import pandas as pd
    from sklearn import preprocessing

    from prince import pca as _prince_pca
    from prince import utils
    from prince.famd import FAMD

    @utils.check_is_dataframe_input
    def _patched_fit(self, X, y=None):
        self.num_cols_ = X.select_dtypes(include=["float"]).columns.tolist()
        if not self.num_cols_:
            raise ValueError("All variables are qualitative: MCA should be used")
        self.cat_cols_ = X.columns.difference(self.num_cols_).tolist()
        if not self.cat_cols_:
            raise ValueError("All variables are quantitative: PCA should be used")

        X_num = X[self.num_cols_].copy()
        self.num_scaler_ = preprocessing.StandardScaler().fit(X_num)
        X_num[:] = self.num_scaler_.transform(X_num)

        X_cat = X[self.cat_cols_]
        self.cat_scaler_ = preprocessing.OneHotEncoder(
            handle_unknown=self.handle_unknown
        ).fit(X_cat)
        X_cat_oh = pd.DataFrame.sparse.from_spmatrix(
            self.cat_scaler_.transform(X_cat),
            index=X_cat.index,
            columns=self.cat_scaler_.get_feature_names_out(self.cat_cols_),
        )
        prop = X_cat_oh.sum() / X_cat_oh.sum().sum() * 2
        X_cat_oh_norm = X_cat_oh.sub(X_cat_oh.mean(axis="rows")).div(
            prop**0.5, axis="columns"
        )
        # Fix: sparse arithmetic sets fill_value=nan; .fillna(0) is required
        X_cat_oh_norm = X_cat_oh_norm.sparse.to_dense().fillna(0)

        Z = pd.concat([X_num, X_cat_oh_norm], axis=1)
        _prince_pca.PCA.fit(self, Z)

        rc = self.row_coordinates(X)
        weights = np.ones(len(X_cat_oh)) / len(X_cat_oh)
        norm = (rc**2).multiply(weights, axis=0).sum()
        eta2 = pd.DataFrame(index=rc.columns)
        for i, col in enumerate(self.cat_cols_):
            tt = X_cat_oh[
                [f"{col}_{cat}" for cat in self.cat_scaler_.categories_[i]]
            ]
            ni = (tt / len(tt)).sum()
            eta2[col] = (
                rc.apply(
                    lambda x: (tt.multiply(x * weights, axis=0).sum() ** 2 / ni).sum()
                )
                / norm
            ).values
        self.column_coordinates_ = pd.concat(
            [self.column_coordinates_.loc[self.num_cols_] ** 2, eta2.T]
        )
        self.column_coordinates_.columns.name = "component"
        self.column_coordinates_.index.name = "variable"
        return self

    FAMD.fit = _patched_fit


def apply_patches():
    _fix_prince_famd()
