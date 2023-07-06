# This utility file should alway go where the env file is.

# For funtion annotations
from typing import Union
from typing import Tuple

# Python libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Mac libraries
import os

# Personal libraries
import env

################### CACHING FILE ##############################################

# Remove encoding while loading csv data to python
def catch_encoding_errors_(fileName:str) -> pd.DataFrame:
    
    """
    parameters:
        fileName: csv file name. Should look like (file.csv)
    return:
        file dataframe with no encoding errors
    """
    
    # list of encodings to check for
    encodings = ['utf-8', 'latin-1', 'cp1252', 'utf-16']
    
    # check encodings and return dataframe
    for encoding in encodings:
        try:
            df = pd.read_csv(fileName, encoding=encoding, index_col=0)
            break
        except UnicodeDecodeError:
            print(f"Failed to decode with {encoding} encoding.")
    return df
        

################### GET NON-EXISTING FILE ##############################################

# get data from codeup data sql database
def get_codeup_sql_data_(db_name: str, table_name: str = None, query: str= None, fileName="UnamedFile") -> Tuple[pd.DataFrame, str]:
    """
    paremeters:
        db_name: name of the database you wich to access
        table_name: name of table you are quering from
        query: (optional argument) the query you want to retrieve
        fileName: name of file (will automaticly save as csv)
        
        note: enter query or table name NOT BOTH

    return:
        data: panads data frame fromt sql query
        query: the query used to retreive the data
    """
    if table_name: # if table is given
        query=f"""
            SELECT *
            FROM {table_name};
            """
        # access the database and retreive the data
        data = pd.read_sql(query, env.get_db_access(db_name))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        data.to_csv(f"{fileName}.csv", mode= "w")

    elif query:
        # access the database and retreive the data
        data = pd.read_sql(query, env.get_db_access(db_name))
    
        # Write that dataframe to disk for later. Called "caching" the data for later.
        data.to_csv(f"{fileName}.csv", mode= "w")

    return data, query # return both the data and the query

################### GET EXISTING FILE ##############################################

# get existing file in current directory
def get_existing_csv_file_(fileName:str, db_name: str= None, table_name: str = None, query: str= None) -> pd.DataFrame:
    """
    parameters:
        fileName: csv file name. Should look like (file)
        db_name: name of the database you wich to access
        table_name: name of table you are quering from
        query: (optional argument) the query you want to retrieve
    return:
        file dataframe with no encoding errors after cheking for existance of file (in current directory)
    """
    if os.path.isfile(f"{fileName}.csv"):
        return catch_encoding_errors_(f"{fileName}.csv")
    else:
        print("Getting data from Codeup database...!")
        data, query= get_codeup_sql_data_(db_name=db_name, query=query, fileName=fileName)
        return data