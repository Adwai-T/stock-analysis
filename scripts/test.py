from pathlib import Path
import pandas as pd
from feature_engineering import add_features

path = Path("./data/allData/CIPLA.csv")
df = pd.read_csv(path)
df["date"] = pd.to_datetime(df["date"])

df_fe = add_features(df)

print(df_fe.head())
print(df_fe.shape)
