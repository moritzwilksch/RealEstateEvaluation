#%%
import polars as pl
import pandas as pd
import json

#%%
with open("data/properties_realEstate.json", "r") as f:
    properties = json.load(f)

#%%
import os
from pymongo import MongoClient
mongo_user = os.getenv("MONGOUSER")
mongo_pass = os.getenv("MONGOPASS")

client = MongoClient(
    f"mongodb://{mongo_user}:{mongo_pass}@157.90.167.200:27017/edgar?authSource=admin"
    # authSource referrs to admin collection in mongo, this needs to be here as a param otherwise: AuthenticationFailed
)
db = client["immoscout"]
coll = db.properties

#%%
def fix_property(property_):
    property_["_id"] = property_["_id"]["$oid"]
    return property_


coll.insert_many([fix_property(prop) for prop in properties])

#%%
# class Listing:
#     def from_json(self, json_data):
#         self.id = json_data.get("_id").get("$oid")
#         self.address = json_data.get("address").get("location")
#         self.city = json_data.get("city")
#         self.zipCode = json_data.get("zipCode")
#         self.quarter = json_data.get("quarter")
#         self.street = json_data.get("street")
#         self.geoLocation = json_data.get("geoLocation")
