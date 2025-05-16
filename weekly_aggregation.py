import pandas as pd
import dotenv

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

# —– 7. Join & hitung Return —–
merged = weekly.join(ihsg, how="inner")
merged = merged.dropna(subset=["Open","Close"])
merged["pct_return"] = (merged["Close"] - merged["Open"]) / merged["Open"] * 100

print("Merged head:")
print(merged[["total","positive","pct_positive","negative", "pct_negative", "neutral", "pct_neutral","Open","Close","pct_return"]])
