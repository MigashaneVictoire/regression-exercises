# For funtion annotations
from binascii import a2b_qp
from typing import Union
from typing import Tuple

# Python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Personal libraries
import acquire_
import env

def wrangle_zillow() -> pd.DataFrame:
    """
    return the prepared 2017 single family data
    """
    # sql query
    query = """
    SELECT bedroomcnt, 
            bathroomcnt,
            calculatedfinishedsquarefeet,
            taxvaluedollarcnt,
            yearbuilt,
            taxamount,
            fips
    FROM properties_2017
    WHERE propertylandusetypeid = 261 -- Single family home
    """

    # get existing csv data from the util directory
    zillow = acquire_.get_existing_csv_file_(fileName ="zillow_single_family")

    # rename dataframe columns
    zillow = zillow.rename(columns={"bedroomcnt":"bedrooms",
                        "bathroomcnt":"bathrooms",
                        "calculatedfinishedsquarefeet":"sqr_feet",
                        "taxvaluedollarcnt":"tax_value",
                        "yearbuilt":"year_built",
                        "taxamount":"tax_amount",
                        "fips":"county"})

    # drop all nulls in the dataframe
    zillow = zillow.dropna()

    # convert data type from float to int
    zillow.bedroomcnt = zillow.bedrooms.astype(int)
    zillow.yearbuilt = zillow.year_built.astype(int)

    # remove the duplocated rows
    zillow = zillow.drop_duplicates(keep="first")
    
    return zillow
# -----------------------------------------------------------------
# Split the data into train, validate and train
def split_data_(df: pd.DataFrame, test_size: float =.2, validate_size: float =.2, stratify_col: str =None, random_state: int=95) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    parameters:
        df: pandas dataframe you wish to split
        test_size: size of your test dataset
        validate_size: size of your validation dataset
        stratify_col: the column to do the stratification on
        random_state: random seed for the data

    return:
        train, validate, test DataFrames
    '''
    # no stratification
    if stratify_col == None:
        # split test data
        train_validate, test = train_test_split(df, 
                                                test_size=test_size, 
                                                random_state=random_state)
        # split validate data
        train, validate = train_test_split(train_validate, 
                                           test_size=validate_size/(1-test_size),
                                            random_state=random_state)
    # stratify split
    else:
        # split test data
        train_validate, test = train_test_split(df,
                                                test_size=test_size,
                                                random_state=random_state, 
                                                stratify=df[stratify_col])
        # split validate data
        train, validate = train_test_split(train_validate, 
                                           test_size=validate_size/(1-test_size),
                                           random_state=random_state, 
                                           stratify=train_validate[stratify_col])
    return train, validate, test