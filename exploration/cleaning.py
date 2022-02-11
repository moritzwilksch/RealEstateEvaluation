import re
import polars as pl
from rich import print
import os
from pymongo import MongoClient
from pandas import json_normalize

mongo_user = os.getenv("MONGO_INITDB_ROOT_USERNAME")
mongo_pass = os.getenv("MONGO_INITDB_ROOT_PASSWORD")


class DatabaseAdapter:
    def __init__(self):
        self.client = MongoClient(
            f"mongodb://{mongo_user}:{mongo_pass}@localhost:27017/edgar?authSource=admin"
        )
        self.db = self.client["immoscout"]
        self.collection = self.db.properties

    def _clean_document(self, doc: dict) -> dict:
        """Cleans document from DB into dataframe-able dict"""
        cleaned_doc = dict()
        return json_normalize(doc)

    def _clean_df(self, data: pl.DataFrame):
        """ Drops irrelevant columns, renames columns, and fixes data types"""
        data = data.drop(
            [
                "_class",
                "hash",
                "imageUrls",
                "numberOfVisits",
                "request.url",
                "request.type",
                "request.method",
                "vendor.companyWideCustomerId",
                "vendor.salutation",
                "vendor.name",
                "vendor.phoneNumber",
                "vendor.company",
                "lastVisitDate.$date",
                "nextVisitDate.$date",
            ]
        )

        data = data.rename(
            {
                col: re.sub("(address\.)(geoLocation\.)?", "", col)
                for col in data.columns
            }
            | {
                "archived.$date": "archived_date",
                "createdDate.$date": "created_date",
            }
        )

        data = data.with_columns(
            [
                pl.col("commercializationType").cast(pl.Categorical),
                pl.col("condition").cast(pl.Categorical),
            ]
        )
        return data

    def dataframe_from_query(self, query, **kwargs) -> pl.DataFrame:
        result_cursor = self.collection.find(query, **kwargs)
        data: pl.DataFrame = pl.from_pandas(json_normalize(list(result_cursor)))
        data = self._clean_df(data)

        return data


if __name__ == "__main__":
    adapter = DatabaseAdapter()
    df = adapter.dataframe_from_query({})
    # print(list(zip(df.columns, df.dtypes)))

    df.to_parquet("data/cleaned.parquet")
    print(len(df))
