# %%
import pandas as pd
import numpy as np

# %%
df = pd.read_csv("../data/ImmoscoutChristian.csv")

# %%


def fix_dtypes(data: pd.DataFrame):
    string_cols = "id location title url object_description".split()
    cat_cols = "object_type private_offer to_rent balcony built_in_kitchen condition elevator energy_efficiency_class garden parking_spot".split()
    # int_cols = "year_of_construction".split()
    float_cols = "rooms year_of_construction sqm".split()

    data = (
        data
        .assign(rooms=data.rooms.fillna(0))
        .assign(rooms=data.rooms.str.extract("(\d|\.)+")[0].replace(".", np.NaN).astype('float16'))
        .assign(year_of_construction=data.year_of_construction.str.extract("(\d{4})")[0].astype("float16"))  # CAUTION: throws aways dirty stuff
        .assign(square_meters=data.square_meters.replace("UNKNOWN", np.NaN))
        .rename({'square_meters': 'sqm'}, axis=1)
    )

    data[string_cols] = data[string_cols].astype("string")
    data[cat_cols] = data[cat_cols].astype("category")
    # data[int_cols] = data[int_cols].astype("int16")

    data[float_cols] = data[float_cols].astype("float32")

    data = data.drop(['created_at', 'balcony', 'bathrooms', 'built_in_kitchen', 'elevator', 'energy_efficiency_class', 'floor', 'garden', 'parking_spot'], axis=1, errors='ignore')

    return data


# %%
df: pd.DataFrame = df.pipe(fix_dtypes)

#%%
df.to_parquet("../data/ISCM.parquet")
