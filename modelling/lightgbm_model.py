# %%
import pathlib
import sys
from typing import List, Tuple
sys.path.append(str(pathlib.Path('..').resolve()))
from lightgbm import LGBMRegressor
from helpers.model_eval import eval_model
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


root_path = "../"

df = pd.read_csv(root_path + 'data/ISListingsBerlinBrb.csv')
#df = df.fillna('NA')


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

for cc in cat_cols:
    X[cc] = X[cc].cat.codes
xtrain, xval, ytrain, yval = train_test_split(X, y, random_state=42)

# %%
reg = LGBMRegressor(objective='MAPE', n_estimators=1000)
reg.fit(xtrain, ytrain, eval_set=(xval, yval), eval_metric='MAPE', verbose=50, early_stopping_rounds=100)

# %%
preds = reg.predict(xval)

# %%
evaldf = eval_model(yval, reg.predict(xval))

#%%
head25 = evaldf.sort_values('ape').head(25).index
pd.concat((xval.loc[head25], yval.loc[head25]), axis=1)

#%%
# Underpriced:
undervalued = evaldf.sort_values('ape').join([df, yval*xval.living_space]).rename({'0': 'price'}, axis=1).drop(['tags', 'creation', 'publish_date', 'energy_certificate'], axis=1)
undervalued.tail(25)


#%%
from lightgbm import plot_importance

plot_importance(reg)

#%%
shap = reg.predict(xval, pred_contrib=True)

#%%
import shap

explainer = shap.Explainer(reg)
shap_values = explainer(xtrain)

#%%
# visualize the first prediction's explanation
shap.plots.beeswarm(shap_values)

#%%
shap.waterfall_plot(shap_values[42])

#%% code
import numpy as np
a = np.array([1,2,3])

#%%
a.all()
#%%

shap.waterfall_plot(shap_values[42])


#%%
from rich.console import Console
c = Console()

#%%
c.print("[white on red] ALERT! [/]")
c.print("[white on green] GOOD! [/]")
