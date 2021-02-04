# RealEstateEvaluation
Real estate evaluation: Modelling rent near my home town.

## Goal
Finding underpriced appartments! Note that underpriced does not equal "cheap": We want to find appartments that are unusuably cheap considering their location, size, and features.


## Data Set
I'll use a small (`n = 1888`) data set of available listings on Germany's number one real estate site (`last updated: Jan 2021`). These are only listings which are...
- publicly listed as of January 25th, 2021
- in the area of Berlin/Brandenburg
- listed in the category "Appartments for Rent"

The whole data set is not publicly available, but looks like the following excerpt:

| idx | balcony   |   barrier_free | builtin_kitchen   |   cellar | city   | creation                      |   energy_certificate | energy_efficiency   | garden   |   guest_toilet | housenr   |      lat |   lift |   listing_id |   living_space |      lng |   number_of_rooms |   postcode |   price | private_offer   | publish_date                  | quarter                         | street        | tags                                                         | title                                                                                         |
|---:|:----------|---------------:|:------------------|---------:|:-------|:------------------------------|---------------------:|:--------------------|:---------|---------------:|:----------|---------:|-------:|-------------:|---------------:|---------:|------------------:|-----------:|--------:|:----------------|:------------------------------|:--------------------------------|:--------------|:-------------------------------------------------------------|:----------------------------------------------------------------------------------------------|
|  0 | False     |            nan | True              |      nan | Berlin | 2019-10-17T10:48:37.000+02:00 |                    1 | D                   | False    |            nan | nan       | nan      |    nan |    114012650 |         123    | nan      |                 2 |      10117 | 5100    | False           | 2019-10-17T10:48:37.000+02:00 | Mitte (Mitte)                   | nan           | Einbauküche,Keller,Aufzug                                    | Hochwertig möbliertes Penthouse mit Belvedere in einzigartiger Lage und Weitblick über Berlin 
|  1 | True      |            nan | True              |      nan | Berlin | 2020-12-02T17:31:02.000+01:00 |                    1 | D                   | False    |            nan | nan       | nan      |    nan |    124856902 |          98.13 | nan      |                 2 |      10117 | 3925    | False           | 2020-12-02T17:31:02.000+01:00 | Mitte (Mitte)                   | nan           | Balkon/Terrasse,Einbauküche,Keller,Gäste-WC,Aufzug,Stufenlos | Stadtresidenz mit edlem Interieur in herrschaftlicher Lage direkt am Berliner Stadtschloss    
|  2 | True      |            nan | True              |      nan | Berlin | 2020-11-20T12:45:34.000+01:00 |                  nan | nan                 | False    |            nan | 6         |  52.5159 |    nan |    124514586 |          77.2  |  13.3333 |                 3 |      10623 | 1872.1  | False           | 2020-11-20T12:45:34.000+01:00 | Charlottenburg (Charlottenburg) | Wegelystrasse | Balkon/Terrasse,Einbauküche,Keller,Aufzug,Stufenlos          | No.1 Charlottenburg - Wohnung zum Erstbezug                                                   

This data set is obviously not ideal: It is a snapshot of a single point in time and could be extended by also learning from historic listings that are not available anymore. It could be extended in size and level of detail (e.g. single appartment exposés). Unfortunately, there is no publicly available API.

## Approach


## Experiments TBD
- Compare neural net with categorical embeddings against net with OHE categories