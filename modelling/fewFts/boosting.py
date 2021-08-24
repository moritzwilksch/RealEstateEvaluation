# %%
import numpy as np
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_parquet("../../data/ISCM.parquet", columns=['object_type', 'price', 'private_offer', 'rooms', 'sqm', 'to_rent', 'zip_code', 'year_of_construction'])
df[['zip_code', 'to_rent', 'private_offer']] = df[['zip_code', 'to_rent', 'private_offer']].astype('category')

# FILTERING
df = (
    df
    .query("rooms < 5 and rooms >= 1")
    .query("price <= 4000 and price > 0")
    .query("sqm <= 200")
    .query("to_rent == True")
    .drop('to_rent', axis=1)
)

# %%
sns.histplot(df.price, bins=25)

# %%
df[['rooms', 'year_of_construction']] = df[['rooms', 'year_of_construction']].astype('category')
xtrain, xval, ytrain, yval = train_test_split(df.drop('price', axis=1), df.price.astype('int'), random_state=42)

# %%
model = LGBMRegressor(n_estimators=10_000, objective='MSE')
model.fit(xtrain, ytrain, eval_set=(xval, yval), eval_metric='MSE', early_stopping_rounds=50)

preds = model.predict(xval)

# %%
# deltas = np.exp(preds) - np.exp(yval.values)
deltas = preds - yval.values
sns.histplot(deltas)
print(np.quantile(deltas, (0.025, 0.975)))

# %%
comparisondf = pd.DataFrame({
    'real': yval.values,
    'pred': preds
})

# %%
evaldf = pd.Series(abs(deltas)).groupby(xval.rooms).agg(['mean', 'count'])
evaldf

# %%


evaldf = pd.Series(deltas).groupby(xval.sqm).agg(['mean', 'count'])
sns.scatterplot(data=evaldf.reset_index(), x='sqm', y='mean')


# evaldf = pd.Series(deltas).groupby(xval.zip_code).agg(['mean', 'count'])
# plt.scatter(evaldf.index, evaldf['mean'])
# sns.scatterplot(data=evaldf.sort_values('mean'), x=evaldf.sort_values('mean').index, y='mean')
