# %%
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_parquet("../../data/ISCM.parquet", columns=['object_type', 'price', 'private_offer', 'rooms', 'sqm', 'to_rent', 'zip_code', 'year_of_construction'])
df[['zip_code', 'to_rent', 'private_offer']] = df[['zip_code', 'to_rent', 'private_offer']].astype('category')

df = df.query("to_rent == True").drop('to_rent', axis=1)

#%%
NUM_COLS = ['price', 'rooms', 'sqm', 'year_of_construction']
CAT_COLS = ['object_type', 'private_offer', 'zip_code']

# %%


def start_pl(data: pd.DataFrame) -> pd.DataFrame:
    return data.copy()


def impute_numeric_cols(data: pd.DataFrame) -> pd.DataFrame:
    si = SimpleImputer(add_indicator=False)
    si.fit(data[NUM_COLS])
    data[NUM_COLS] = si.transform(data[NUM_COLS])
    return data


clean = (
    df
    .pipe(start_pl)
    .pipe(impute_numeric_cols)
)


# %%
df.private_offer.unique()
# %%

xtrain, xval, ytrain, yval = train_test_split(df.drop('price', axis=1), df.price, random_state=42)

# %%
model = LGBMRegressor(objective='MAE')
model.fit(xtrain, ytrain, eval_set=(xval, yval), eval_metric='MAPE')
