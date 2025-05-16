import pandas as pd
import numpy as np
import dotenv
from sklearn.linear_model import LogisticRegression

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

print(merged[["total","pct_positive","pct_negative","pct_neutral","Return"]])

df = merged
df[["pct_positive","pct_negative","pct_neutral"]].plot(title="Sentimen Publik per Hari")
df["Return"].plot(secondary_y=True, title="IHSG Return vs Sentimen")

for lbl in ["pct_positive","pct_negative","pct_neutral"]:
    print(lbl, df[lbl].corr(df["Return"].shift(-1)))

df["up"] = (df["Return"] > 0).astype(int)
logit = LogisticRegression().fit(df[["pct_positive","pct_negative"]], df["up"])