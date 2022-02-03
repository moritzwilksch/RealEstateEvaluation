#%%
import polars as pl

df = pl.read_parquet("data/cleaned.parquet")

#%%
df.head().to_pandas()

#%%
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x=df.commercializationType.to_numpy())