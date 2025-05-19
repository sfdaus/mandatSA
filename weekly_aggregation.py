import numpy as np
import pandas as pd
import dotenv
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import LeaveOneOut
import statsmodels.api as sm
from scipy.stats import spearmanr

# —– 1. Siapkan sentiment weekly (kamu sudah punya) —–
config = dotenv.dotenv_values(".env")
df = pd.read_csv(config["FILE_LABELED"], parse_dates=["created_at"])
df["date"] = df["created_at"].dt.date
df = df.set_index(pd.to_datetime(df["date"]))
daily = df.groupby([pd.Grouper(freq="D"), "label"]).size().unstack(fill_value=0)
weekly = daily.resample("W").sum()
weekly.index = weekly.index + pd.Timedelta(days=1)
weekly["total"] = weekly.sum(axis=1)
for lbl in ["negative","neutral","positive"]:
    weekly[f"pct_{lbl}"] = weekly[lbl]/weekly["total"]*100
weekly = weekly[[
    "total","negative","pct_negative",
    "neutral","pct_neutral",
    "positive","pct_positive"
]]

# —– 2. Baca IHSG weekly CSV —–
ihsg = pd.read_csv(config["FILE_IHSG"])

# —– 3. Ekstrak tanggal awal minggu (DD/MM/YYYY) via regex —–
ihsg["Week_Start_Str"] = ihsg["Tanggal"].str.extract(r'(\d{2}/\d{2}/\d{4})')[0]

# —– 4. Parse ke datetime dengan format spesifik —–
ihsg["Week_Start"] = pd.to_datetime(
    ihsg["Week_Start_Str"], 
    format="%d/%m/%Y",
    errors="coerce"
)

# —– 5. Buang baris yang gagal parse —–
ihsg = ihsg.dropna(subset=["Week_Start"])
# Set index dan pilih kolom pasar
ihsg = ihsg.set_index("Week_Start")[["Open","High","Low","Close","Volume"]].sort_index()

# —– 6. Convert ke numeric —–
for col in ["Open","High","Low","Close","Volume"]:
    ihsg[col] = pd.to_numeric(ihsg[col], errors="coerce")

# —– 7. Join & hitung Return, drop tgl 31 maret karena lebaran IHSG tutup seminggu —–
weekly = weekly.drop(index=pd.to_datetime("2025-03-31"))
merged = weekly.join(ihsg, how="inner")
merged = merged.dropna(subset=["Open","Close"])
merged["pct_return"] = (merged["Close"] - merged["Open"]) / merged["Open"] * 100


# print(merged[["total","positive","pct_positive","negative", "pct_negative", "neutral", "pct_neutral","Open","Close","pct_return"]])

# Ringkasan statistik untuk persentase sentimen dan return
desc = merged[['pct_positive','pct_negative','pct_neutral','pct_return']].describe().T
# print(desc[['mean','std','min','max']])

# Matriks korelasi Pearson
corr_mat = merged[['pct_positive','pct_negative','pct_neutral','pct_return']].corr()
# print(corr_mat)

# Khusus:
# print("Corr %pos vs %return:",  corr_mat.loc['pct_positive','pct_return'])
# print("Corr %neg vs %return:",  corr_mat.loc['pct_negative','pct_return'])
# print("Corr %neu vs %return:",  corr_mat.loc['pct_neutral','pct_return'])

# Pilih dua kategori untuk hindari multikol:
X = merged[['pct_positive','pct_negative']]  
X = sm.add_constant(X)
y = merged['pct_return']

model = sm.OLS(y, X).fit()
# print(model.summary())
pearson_neg = merged['pct_negative'].corr(merged['pct_return'])
rho_neg, pval_neg = spearmanr(merged['pct_negative'], merged['pct_return'])
# print(f"Negatif → Pearson r = {pearson_neg:.3f}, Spearman ρ = {rho_neg:.3f} (p = {pval_neg:.3f})")

pearson_pos = merged['pct_positive'].corr(merged['pct_return'])
rho_pos, pval_pos = spearmanr(merged['pct_positive'], merged['pct_return'])
# print(f"Positif → Pearson r = {pearson_pos:.3f}, Spearman ρ = {rho_pos:.3f} (p = {pval_pos:.3f})")

pearson_neut = merged['pct_neutral'].corr(merged['pct_return'])
rho_neut, pval_neut = spearmanr(merged['pct_neutral'], merged['pct_return'])
# print(f"Neutral → Pearson r = {pearson_neut:.3f}, Spearman ρ = {rho_neut:.3f} (p = {pval_neut:.3f})")

#=================================== versi balance

merged['balance'] = merged['pct_positive'] - merged['pct_negative']

# 1. Spearman correlation
rho, pval = spearmanr(merged['balance'], merged['pct_return'])
# print(f"Spearman rho={rho:.3f}, p={pval:.3f}")

obs_rho, _ = spearmanr(merged['balance'], merged['pct_return'])
B = 5000
count = 0
for _ in range(B):
    y_perm = np.random.permutation(merged['pct_return'])
    rho_perm, _ = spearmanr(merged['balance'], y_perm)
    if abs(rho_perm) >= abs(obs_rho):
        count += 1
p_perm = count / B
# print(f"Observed ρ = {obs_rho:.3f}")
# print(f"Permutation p-value = {p_perm:.3f}")

X = merged[['balance']].values
y = merged['pct_return'].values

huber = HuberRegressor().fit(X, y)
# print("Huber coef (balance):", huber.coef_[0])
# print("Huber intercept:", huber.intercept_)

coefs = []
for _ in range(2000):
    samp = merged.sample(frac=1, replace=True)
    Xs = samp[['balance']].values
    ys = samp['pct_return'].values
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
# ols = sm.OLS(merged['pct_return'], X2, missing='drop').fit(cov_type='HC3')
# print(ols.summary())

y_pred_full = sm.OLS(y, X2).fit().predict(X2)
from scipy.stats import spearmanr
rho, pval = spearmanr(y_pred_full, y)
print(f"Spearman rho(pred vs actual) = {rho:.3f} (p = {pval:.3f})")