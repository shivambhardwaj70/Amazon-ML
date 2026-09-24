import lightgbm as lgb

def make_lgb_regressor(params=None):
    d = dict(
        objective="regression", metric="rmse",
        n_estimators=5000, learning_rate=0.02,
        num_leaves=127, subsample=0.8, colsample_bytree=0.6,
        reg_lambda=5.0, min_child_samples=50, n_jobs=-1, random_state=42,
    )
    if params:
        d.update(params)

    def fit(Xtr, ytr, Xval, yval):
        m = lgb.LGBMRegressor(**d)
        m.fit(
            Xtr, ytr, eval_set=[(Xval, yval)],
            callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)],
        )
        return m.predict(Xval)

    return fit

# RMSLE metric -> fit on np.log1p(y), predict, then np.expm1(pred).
