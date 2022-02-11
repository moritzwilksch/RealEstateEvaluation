#%%
import secrets

DB_URL = secrets.DB_URL

#%%
from sqlalchemy import create_engine

engine = create_engine(DB_URL)

#%%
query = """ 

WITH important_vars AS (
    SELECT prop.id,
           created_at,
           location,
           object_type,
           price,
           private_offer,
           rooms,
           square_meters,
           title,
           to_rent,
           url,
           zip_code
           /*
           epa.parking_spot,
           epa.year_of_construction,
           epa.balcony,
           epa.bathrooms,
           epa.built_in_kitchen,
           epa.condition,
           epa.elevator,
           epa.energy_efficiency_class,
           epa.floor,
           epa.garden,
           epa.object_description
           */
    FROM public.property prop
             LEFT JOIN extended_property ep on ep.id = prop.extended_property_id
             LEFT JOIN public.extended_property_attributes epa ON ep.attributes_id = epa.id
             LEFT JOIN public.address ON prop.address_id = address.id
    WHERE prop.country_code = 'DE'
      AND address.city = 'Berlin'
      AND prop.commercialization_type = 'RESIDENTIAL'
)

SELECT *
FROM important_vars;
-- LIMIT 10;

"""
#%%
import pandas as pd

df_from_db = pd.read_sql(query, engine, index_col="id").reset_index()

#%%
df = df_from_db.copy()
dtypes = {
    # "id": "category",
    # "created_at": "datetime64",
    "location": "string",
    "object_type": "category",
    "price": "int",
    "private_offer": "bool",
    "rooms": "category",
    "square_meters": "int",
    "title": "string",
    "to_rent": "bool",
    "url": "string",
    "zip_code": "category",
}

df = df.assign(price=df.price.replace("", 0).fillna(0)).assign(
    square_meters=df.square_meters.replace("", 0).fillna(0)
)

for col in dtypes.keys():
    print(f"[INFO] Converting {col}")
    df[col] = df[col].astype(dtypes[col])

df = df.drop('id', axis=1)

#%%
df.to_parquet("data/berlin.parquet")

#%%
# df.to_parquet("data/berlin_raw.parquet")
# df.to_csv("data/berlin_raw.csv")
