import os
import pandas as pd
from databricks import sql
import time

SERVER_HOSTNAME = os.getenv("SERVER_HOSTNAME")
HTTP_PATH = os.getenv("HTTP_PATH")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

def execute_query(dbsql_query):
    """
    Fetches data from the Databricks database and returns it as a pandas dataframe

    Returns
    -------
    df : pandas dataframe
        basic query of data from Databricks as a pandas dataframe
    """
    conn = sql.connect(
        server_hostname=SERVER_HOSTNAME,
        http_path=HTTP_PATH,
        access_token=ACCESS_TOKEN,
    )
    cursor = conn.cursor()
    cursor.execute(dbsql_query)
    df = cursor.fetchall_arrow().to_pandas()
    cursor.close()
    conn.close()
    return df


def get_expenditure_w_porverty_by_country_year():
    start = time.time()
    df = execute_query("""
        SELECT *
        FROM boost.pov_expenditure_by_country_year
    """)
    df['decentralized_expenditure'].fillna(0, inplace=True)
    end = time.time()
    print('get_expenditure_w_porverty_by_country_year', end-start )
    return df

#TODO add the filter by the years
def get_expenditure_by_country_func_year():
    start = time.time()
    query = '''
        SELECT *
        FROM boost.expenditure_by_country_func_year
    '''
    end = time.time()
    df = execute_query(query)
    print("get_expenditure_by_country_func_year", end-start)
    return df

#TODO add the filter by the years
def get_edu_private_expenditure():
    start = time.time()
    query = '''
        SELECT country_name, year, real_expenditure
        FROM boost.edu_private_expenditure_by_country_year
        ORDER BY country_name, year
    '''
    df = execute_query(query)
    end = time.time()
    print("get_edu_private_expenditure", end-start)
    return df

# The full dataset is big therefore requiring the list of countries for filtering
def get_hd_index(countries):
    start =time.time()
    query= '''
        SELECT * FROM indicator.global_data_lab_hd_index
    '''
    country_list = "', '".join(countries)
    query += f" WHERE country_name IN ('{country_list}')"
    query += ' ORDER BY country_name, year'
    end = time.time()
    df =  execute_query(query)
    print(" get_hd_index", end-start)
    return df

def get_learning_poverty_rate():
    start = time.time()
    query = '''
        SELECT * FROM indicator.learning_poverty_rate
    '''
    df = execute_query(query)
    end = time.time()
    print("get_learning_poverty_rate", end - start)
    return df

def get_expenditure_by_country_func_econ_year():
    start = time.time()
    query = """
        SELECT * FROM boost.expenditure_by_country_func_econ_year
    """
    df = execute_query(query)
    end = time.time()
    print("get_expenditure_by_country_func_econ_year", end-start)
    return df
