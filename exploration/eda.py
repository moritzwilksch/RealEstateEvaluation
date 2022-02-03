#%%
import polars as pl
import pandas as pd
# df = pl.read_parquet("data/cleaned.parquet")
df = pd.read_parquet("data/cleaned.parquet")

#%%
df.head()

#%%
# df = df.with_columns([
#     pl.col("created_date").str.strptime(pl.Datetime, fmt="%Y-%m-%dT%H:%M:%S.%fZ"),
#     pl.col("archived_date").str.strptime(pl.Datetime, fmt="%Y-%m-%dT%H:%M:%S.%fZ"),
#     ])

#%%
df = df.assign(
    created_date=df.created_date.astype("datetime64"),
    archived_date=df.archived_date.astype("datetime64"),
    duration=lambda d: (d["archived_date"]  -d["created_date"])
)
#%%
import seaborn as sns
import matplotlib.pyplot as plt
sns.displot(df["duration"].dt.days, bins=50)

#%%
df.duration.isnull().mean()