# %%
import enum
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pathlib
import sys
from typing import List, Tuple
sys.path.append(str(pathlib.Path('..').resolve()))
from helpers.model_eval import eval_model


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

oe = OrdinalEncoder()
oe.fit(X[[col for col in cat_cols if col not in bool_cols]])
xtrain[[col for col in cat_cols if col not in bool_cols]] = oe.transform(xtrain[[col for col in cat_cols if col not in bool_cols]])
xval[[col for col in cat_cols if col not in bool_cols]] = oe.transform(xval[[col for col in cat_cols if col not in bool_cols]])

# %%
scale_cols = xtrain.drop(cat_cols, axis=1).columns

ss = StandardScaler()
xtrain[scale_cols] = ss.fit_transform(xtrain[scale_cols])
xval[scale_cols] = ss.fit_transform(xval[scale_cols])

# %%
xtrain_num = xtrain[scale_cols].values.astype(np.float32)
xval_num = xval[scale_cols].values.astype(np.float32)
xtrain_cat = xtrain[cat_cols].values.astype(np.float32)
xval_cat = xval[cat_cols].values.astype(np.float32)


# %%
cardinalities = {
    "postcode": 189,
    "quarter": 96,
    "city": 2,
    "balcony": 2,
    "barrier_free": 0,
    "builtin_kitchen": 2,
    "cellar": 0,
    "garden": 2,
    "guest_toilet": 0,
    "lift": 0,
    "street": 813,
    "energy_certificate": 1,
    "energy_efficiency": 8
}

emb_dims = {'postcode': 20,
            'quarter': 10,
            'city': 2,
            'balcony': 2,
            'barrier_free': 1,
            'builtin_kitchen': 1,
            'cellar': 1,
            'garden': 1,
            'guest_toilet': 1,
            'lift': 1,
            'street': 20,
            'energy_certificate': 1,
            'energy_efficiency': 10,
            }


#%%

num_in = tf.keras.Input(shape=(len(scale_cols, )))
emb_in = tf.keras.Input(shape=(len(emb_dims), ))

embedding_layers = [tf.keras.layers.Embedding(input_dim=cardinalities[c]+1, output_dim=emb_dims[c]) for c in cat_cols]
embeddings = [emb(emb_in[:, i]) for i, emb in enumerate(embedding_layers)]
concat = [tf.keras.layers.Flatten()(e) for e in embeddings]
concat = tf.keras.layers.Concatenate()(embeddings)

dense1 = tf.keras.layers.Dense(units=50, activation='relu', kernel_regularizer=tf.keras.regularizers.l2())(concat)

out = tf.keras.layers.Dense(units=1, activation='linear')(dense1)

model = tf.keras.Model(inputs=[num_in, emb_in], outputs=[out])
model.compile('adam', 'mean_absolute_percentage_error', metrics=['mean_absolute_percentage_error'])

#%%
import time
checkpointer = tf.keras.callbacks.ModelCheckpoint(root_path + ".modelcheckpoints", save_best_only=True, save_weights_only=True)
tic = time.time()
hist = model.fit([xtrain_num, xtrain_cat], ytrain, validation_data=([xval_num, xval_cat], yval), epochs=40, callbacks=[checkpointer])
tac = time.time()
print(f"Fit took {tac - tic:.1f} seconds!")
#%%
model.load_weights(root_path + ".modelcheckpoints")
#model.load_weights(root_path + ".modelcheckpoints")
#%%
pd.DataFrame(hist.history)[['loss', 'val_loss']].plot()

#%%
eval_model(yval, model.predict([xval_num, xval_cat]).flatten())
# %%
# TODO: Net using One Hot Encoded inputs
