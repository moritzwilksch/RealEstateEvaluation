# %%
from random import random
import pandas as pd
import numpy as np

# %%
df = pd.read_csv("../data/ImmoscoutChristian.csv")

# %%


def fix_dtypes(data: pd.DataFrame):
    string_cols = "id location title url object_description".split()
    cat_cols = "object_type private_offer to_rent balcony built_in_kitchen condition elevator energy_efficiency_class garden parking_spot zip_code".split()
    # int_cols = "year_of_construction".split()
    float_cols = "sqm".split()
    bool_cols = "private_offer to_rent".split()

    # data = data.sample(10_000, random_state=42)  # CAUTION!

    data = (
        data
        .assign(rooms=data.rooms.fillna(np.NaN))
        .assign(rooms=data.rooms.str.extract("(\d|\.)+")[0].replace(".", np.NaN).astype('float').astype('UInt8'))
        .assign(year_of_construction=data.year_of_construction.str.extract("(\d{4})")[0].astype('float').astype('UInt16'))  # CAUTION: throws aways dirty stuff
        .assign(square_meters=data.square_meters.replace("UNKNOWN", np.NaN))
        .assign(price=(df['price']/100).round(0).astype('Int32'))
        .rename({'square_meters': 'sqm'}, axis=1)
    )

    df[bool_cols] = df[bool_cols].astype('boolean')
    data[string_cols] = data[string_cols].astype("string")
    data[cat_cols] = data[cat_cols].astype("category")
    # data[int_cols] = data[int_cols].astype("int16")

    data[float_cols] = data[float_cols].astype("float32")

    data['sqm'] = data.sqm / 100.0

    data = (
        data
        .drop(['created_at', 'balcony', 'bathrooms', 'built_in_kitchen', 'elevator', 'energy_efficiency_class', 'floor', 'garden', 'parking_spot'], axis=1, errors='ignore')
        .loc[data['object_type'].isin(('HOUSE', 'APARTMENT', np.nan))]  # FILTER!
        )

    return data


#%%
clean: pd.DataFrame = df.pipe(fix_dtypes)

#%%
clean.to_parquet("../data/ISCM.parquet")
