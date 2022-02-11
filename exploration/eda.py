#%%
import polars as pl
import pandas as pd
from rich import print

#%%

df = pl.read_parquet("data/cleaned.parquet")
# df = pd.read_parquet("data/cleaned.parquet")

#%%
df.head()

#%%
df = df.with_columns(
    [
        pl.col("created_date").str.strptime(pl.Datetime, fmt="%Y-%m-%dT%H:%M:%S.%fZ"),
        pl.col("archived_date").str.strptime(pl.Datetime, fmt="%Y-%m-%dT%H:%M:%S.%fZ"),
    ]
)

#%%
# df = df.assign(
#     created_date=df.created_date.astype("datetime64"),
#     archived_date=df.archived_date.astype("datetime64"),
#     duration=lambda d: (d["archived_date"] - d["created_date"]),
# )

cat_cols = [
    "objectType",
    "source",
    "city",
    "zipCode",
    "quarter",
]
df = df.with_columns(
    [
        (pl.col("archived_date") - pl.col("created_date")).alias("online_duration"),
        pl.col(cat_cols).cast(pl.Categorical),
    ]
)
#%%
import datetime
df.filter(pl.col("online_duration").is_not_null()).select("online_duration")


#%%
for col in df.columns:
    if df[col].n_unique() < 20 and df[col].dtype != pl.Boolean:
        print(df[col].value_counts().head(10))

#%%
titles = df["title"]#.to_pandas()

#%%
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(ngram_range=(2, 5), analyzer="char_wb")
vectors = cv.fit_transform(titles.to_list())

#%%
from sklearn.metrics.pairwise import cosine_similarity

query_text = titles.sample(1).to_list()
query_vector = cv.transform(query_text)

print(query_text)
sims = cosine_similarity(query_vector, vectors)
top_matches = titles.to_frame()[sims.argsort()[0][-15:], :]["title"].to_list()
for idx, m in enumerate(top_matches):
    print(f"{idx:>3}: {m}")
