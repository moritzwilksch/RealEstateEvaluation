#%%
import polars as pl

# df = pl.DataFrame(
#     {
#         "start": ["2022-01-01 14:37:21", "2022-01-05 18:54:21"],
#         "end": ["2022-01-02 01:34:21", "2022-01-06 05:54:21"],
#     }
# )


# df = df.with_columns(pl.col("*").str.strptime(pl.Datetime))

# print(df.with_column((pl.col("end") - pl.col("start")).alias("duration")))



import seaborn as sns
df = pl.from_pandas(sns.load_dataset("tips"))

#%%
sns.scatterplot(data=df.to_pandas(), x="total_bill", y="tip")