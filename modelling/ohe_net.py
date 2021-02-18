#%%
import sys
import pathlib
sys.path.append(str(pathlib.Path('..').resolve()))

# %%
from helpers.model_eval import eval_model
import enum
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from typing import List, Tuple




root_path = "../"

df = pd.read_csv(root_path + 'data/ISListingsBerlinBrb.csv')


# %%
cat_cols = ['postcode', 'quarter', 'city', 'balcony', 'barrier_free', 'builtin_kitchen', 'cellar', 'garden',
            'guest_toilet', 'lift', 'street', 'energy_certificate', 'energy_efficiency']

bool_cols = ['balcony', 'builtin_kitchen', 'garden', 'private_offer']

X = df.drop('title tags publish_date price creation housenr listing_id'.split(), axis=1)
y = df.price/(df.living_space+1)

add_no_cat = 'barrier_free cellar energy_efficiency guest_toilet'.split()
X["energy_certificate"] = X["energy_certificate"].fillna(False)
X[add_no_cat] = X[add_no_cat].fillna('NO')
X[cat_cols] = X[cat_cols].fillna('NA')
numcols = X.loc[:, ~X.columns.isin(cat_cols)].columns

X[cat_cols] = X[cat_cols].astype('category')
xtrain, xval, ytrain, yval = train_test_split(X, y, random_state=42, test_size=0.2)

# %%

ohe = OneHotEncoder(handle_unknown='ignore')

train_ohe = ohe.fit_transform(xtrain[cat_cols]).toarray().astype(np.int8)
val_ohe = ohe.transform(xval[cat_cols]).toarray().astype(np.int8)

xtrain_ohe = pd.concat((xtrain.drop(cat_cols, axis=1).fillna(0), pd.DataFrame(train_ohe, index=xtrain.index)), axis=1).astype(np.float)
xval_ohe = pd.concat((xval.drop(cat_cols, axis=1).fillna(0), pd.DataFrame(val_ohe, index=xval.index)), axis=1).astype(np.float)




# %%
scale_cols = xtrain.drop(cat_cols, axis=1).columns

ss = StandardScaler()
xtrain[scale_cols] = ss.fit_transform(xtrain[scale_cols])
xval[scale_cols] = ss.fit_transform(xval[scale_cols])


#%%

model = tf.keras.Sequential([
    #tf.keras.layers.Dense(units=500, activation='relu'),
    tf.keras.layers.Dense(units=50, activation='relu', kernel_regularizer=tf.keras.regularizers.l2()),
    tf.keras.layers.Dense(units=1, activation='linear'),
])

model.compile('adam', 'mean_squared_error', metrics=['mean_absolute_percentage_error'])

#%%
import time
checkpointer = tf.keras.callbacks.ModelCheckpoint(root_path + ".modelcheckpoints", save_best_only=True, save_weights_only=True)
tic = time.time()
hist = model.fit(xtrain_ohe.values, ytrain, validation_data=(xval_ohe.values, yval), epochs=40, callbacks=[checkpointer])
tac = time.time()
print(f"Fit took {tac - tic:.1f} seconds!")
#%%
model.load_weights(root_path + ".modelcheckpoints")
#model.load_weights(root_path + ".modelcheckpoints")
#%%
pd.DataFrame(hist.history)[['loss', 'val_loss']].plot()

#%%
eval_model(yval, model.predict(xval_ohe.values).flatten())
