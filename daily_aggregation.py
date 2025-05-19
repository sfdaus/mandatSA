import pandas as pd
import numpy as np
import dotenv
from sklearn.isotonic import spearmanr
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.model_selection import LeaveOneOut
import statsmodels.api as sm

# —– 0. Load paths —–
config     = dotenv.dotenv_values(".env")
tweets_csv = config["FILE_LABELED"]     # e.g. "labeling_manual.csv"
ihsg_csv   = config["FILE_IHSG_DAILY"]  # e.g. "ihsg_daily.csv"

# —– 1. Load & prepare IHSG harian —–
ihsg = pd.read_csv(ihsg_csv)

# 1a. Autodetect and parse date column
date_col = next((c for c in ihsg.columns if c.lower() in ("date","tanggal")), None)
ihsg[date_col] = pd.to_datetime(ihsg[date_col], dayfirst=True, errors="coerce")
ihsg = ihsg.rename(columns={date_col:"Date"}).set_index("Date").sort_index()

# 1b. Drop rows where Date parsing failed
ihsg = ihsg[~ihsg.index.isna()]

# 1c. Normalize timezone (make tz-naive)
if ihsg.index.tz is not None:
    ihsg.index = ihsg.index.tz_localize(None)

# 1d. Convert price/volume columns to numeric & drop non-trading days
for col in ["Open","High","Low","Close","Volume"]:
    if col in ihsg.columns:
        ihsg[col] = pd.to_numeric(ihsg[col].astype(str).str.replace(",",""), errors="coerce")
ihsg = ihsg.dropna(subset=["Close"])
ihsg["Return"] = ihsg["Close"].pct_change() * 100

# 1e. Create sorted array of trading days (tz-naive)
trading_days = np.array(sorted(ihsg.index.unique()))

# —– 2. Load & aggregate tweets per day —–
tweets = pd.read_csv(tweets_csv, parse_dates=["created_at"])
tweets["date"] = tweets["created_at"].dt.normalize()

daily = tweets.groupby(["date","label"]).size().unstack(fill_value=0)
daily["total"] = daily.sum(axis=1)
for lbl in ["positive","negative","neutral"]:
    daily[f"pct_{lbl}"] = daily.get(lbl,0) / daily["total"] * 100

# Ensure index is datetime and tz-naive
daily.index = pd.to_datetime(daily.index, errors="coerce")
daily = daily[~daily.index.isna()]
if daily.index.tz is not None:
    daily.index = daily.index.tz_localize(None)

# —– 3. Map to next trading day via searchsorted —–
def next_trading_day(dt):
    # dt is tz-naive
    if dt in trading_days:
        return dt
    idx = np.searchsorted(trading_days, dt)
    return trading_days[idx] if idx < len(trading_days) else trading_days[-1]

daily["biz_day"] = daily.index.map(next_trading_day)

# —– 4. Aggregate by biz_day —–
tweets_biz = daily.groupby("biz_day").agg({
    "total":   "sum",
    "positive":"sum",
    "negative":"sum",
    "neutral": "sum"
})
for lbl in ["positive","negative","neutral"]:
    tweets_biz[f"pct_{lbl}"] = tweets_biz[lbl] / tweets_biz["total"] * 100

# —– 5. Join with IHSG and display —–
merged = tweets_biz.join(ihsg["Return"], how="inner").dropna(subset=["Return"])

# print(merged[["total","pct_positive","pct_negative","pct_neutral","Return"]])

df = merged
df[["pct_positive","pct_negative","pct_neutral"]].plot(title="Sentimen Publik per Hari")
df["Return"].plot(secondary_y=True, title="IHSG Return vs Sentimen")

# for lbl in ["pct_positive","pct_negative","pct_neutral"]:
#     print(lbl, df[lbl].corr(df["Return"].shift(-1)))

df["up"] = (df["Return"] > 0).astype(int)
logit = LogisticRegression().fit(df[["pct_positive","pct_negative"]], df["up"])

pearson_neg = merged['pct_negative'].corr(merged['Return'])
rho_neg, pval_neg = spearmanr(merged['pct_negative'], merged['Return'])
# print(f"Negatif → Pearson r = {pearson_neg:.3f}, Spearman ρ = {rho_neg:.3f} (p = {pval_neg:.3f})")

pearson_pos = merged['pct_positive'].corr(merged['Return'])
rho_pos, pval_pos = spearmanr(merged['pct_positive'], merged['Return'])
# print(f"Positif → Pearson r = {pearson_pos:.3f}, Spearman ρ = {rho_pos:.3f} (p = {pval_pos:.3f})")

pearson_neut = merged['pct_neutral'].corr(merged['Return'])
rho_neut, pval_neut = spearmanr(merged['pct_neutral'], merged['Return'])
# print(f"Neutral → Pearson r = {pearson_neut:.3f}, Spearman ρ = {rho_neut:.3f} (p = {pval_neut:.3f})")

#=================================== versi balance

merged['balance'] = merged['pct_positive'] - merged['pct_negative']

# 1. Spearman correlation
rho, pval = spearmanr(merged['balance'], merged['Return'])
# print(f"Spearman rho={rho:.3f}, p={pval:.3f}")

obs_rho, _ = spearmanr(merged['balance'], merged['Return'])
B = 5000
count = 0
for _ in range(B):
    y_perm = np.random.permutation(merged['Return'])
    rho_perm, _ = spearmanr(merged['balance'], y_perm)
    if abs(rho_perm) >= abs(obs_rho):
        count += 1
p_perm = count / B
# print(f"Observed ρ = {obs_rho:.3f}")
# print(f"Permutation p-value = {p_perm:.3f}")

X = merged[['balance']].values
y = merged['Return'].values

huber = HuberRegressor().fit(X, y)
# print("Huber coef (balance):", huber.coef_[0])
# print("Huber intercept:", huber.intercept_)

coefs = []
for _ in range(2000):
    samp = merged.sample(frac=1, replace=True)
    Xs = samp[['balance']].values
    ys = samp['Return'].values
    m = HuberRegressor().fit(Xs, ys)
    coefs.append(m.coef_[0])
ci = np.percentile(coefs, [2.5, 97.5])
# print(f"95% Bootstrap CI: [{ci[0]:.3f}, {ci[1]:.3f}]")

loo = LeaveOneOut()
errors = []
for train_i, test_i in loo.split(X):
    X_train, X_test = X[train_i], X[test_i]
    y_train, y_test = y[train_i], y[test_i]
    m = HuberRegressor().fit(X_train, y_train)
    y_pred = m.predict(X_test)
    errors.append((y_test - y_pred)**2)
rmse = np.sqrt(np.mean(errors))
print(f"LOOCV RMSE: {rmse:.3f}")

# y_pred_full = huber.predict(X)
# rho_pred, pval_pred = spearmanr(y_pred_full, y)
# print(f"Spearman ρ (pred vs actual) = {rho_pred:.3f} (p = {pval_pred:.3f})")


X2 = sm.add_constant(merged['balance'])
ols = sm.OLS(merged['Return'], X2, missing='drop').fit(cov_type='HC3')
# print(ols.summary())

y_pred_full = sm.OLS(y, X2).fit().predict(X2)
from scipy.stats import spearmanr
rho, pval = spearmanr(y_pred_full, y)
print(f"Spearman rho(pred vs actual) = {rho:.3f} (p = {pval:.3f})")