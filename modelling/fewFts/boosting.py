#%%
import pandas as pd

df = pd.read_parquet("../../data/ISCM.parquet")

#%%
df = df[['object_type', 'price', 'private_offer', 'rooms', 'sqm', 'to_rent', 'zip_code', 'year_of_construction']]

bool_cols = "private_offer to_rent".split()
df[bool_cols] = df[bool_cols].astype('boolean')

#%%
from sklearn.model_selection import train_test_split

xtrain, ytrain, xval, yval = train_test_split(df.drop('price', axis=1), df.price, random_state=42)

#%%
