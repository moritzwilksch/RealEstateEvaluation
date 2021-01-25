# %%
import pathlib
import sys
from typing import List, Tuple
sys.path.append(str(pathlib.Path('..').resolve()))

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


from helpers.model_eval import eval_model

root_path = "../"

df = pd.read_csv(root_path + 'data/ISListingsBerlinBrb.csv')
#df = df.fillna('NA')


# %%
from catboost import CatBoostRegressor

df['tags'] = df['tags'].astype('string')
def extract_tags(data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    tags = df.tags.split(',', expand=True)
    
#%%
cat_cols = ['postcode', 'quarter', 'city', 'balcony', 'barrier_free', 'builtin_kitchen', 'cellar', 'garden',
            'guest_toilet', 'lift', 'street', 'energy_certificate', 'energy_efficiency']

X = df.drop('title tags publish_date price creation housenr listing_id'.split(), axis=1)
y = df.price/(df.living_space+1)

add_no_cat = 'barrier_free cellar energy_certificate energy_efficiency guest_toilet'.split()
X[add_no_cat] = X[add_no_cat].fillna('NO')
X[cat_cols] = X[cat_cols].fillna('NA')
numcols = X.loc[:, ~X.columns.isin(cat_cols)].columns

X[cat_cols] = X[cat_cols].astype('category')
xtrain, xval, ytrain, yval = train_test_split(X, y, random_state=42)

# %%
cbr = CatBoostRegressor(learning_rate=0.1, iterations=2000, cat_features=cat_cols, loss_function='MAPE', verbose=50)
cbr.fit(xtrain, ytrain, eval_set=(xval, yval))

# %%
preds = cbr.predict(xval)

# %%
evaldf = eval_model(yval, cbr.predict(xval))

#%%
head25 = evaldf.sort_values('ape').head(25).index
pd.concat((xval.loc[head25], yval.loc[head25]), axis=1)

#%%
# Underpriced:
undervalued = evaldf.sort_values('ape').join([df, yval*xval.living_space]).rename({0: 'price'}, axis=1).drop(['tags', 'creation', 'publish_date', 'energy_certificate'], axis=1)
undervalued.tail(25)

