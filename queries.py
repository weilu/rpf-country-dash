import os
import pandas as pd
from databricks import sql

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
    df = execute_query("""
        SELECT e.*, p.poor215
        FROM boost.expenditure_by_country_year e
        LEFT JOIN indicator.poverty p
          ON e.country_name = p.country_name AND e.year = p.year
        ORDER BY e.country_name, e.year
    """)
    df['decentralized_expenditure'].fillna(0, inplace=True)
    ## just for testing (per capital is missing, underlying database changed?)
    df["per_capita_real_expenditure"] = df.expenditure
    df["per_capita_expenditure"] = df.expenditure
    return df

#TODO add the filter by the years
def get_expenditure_by_func_country_year():
    query = 'SELECT l.country_name, l.year, l.admin0, l.func, l.real_expenditure '\
        'FROM boost.expenditure_by_country_admin_func_sub_econ_sub_year as l  '\
        'WHERE l.func in ("Education", "Health")'
    df = execute_query(query)
    return df

#TODO add the filter by the years
def get_edu_private_expenditure():
    query = 'SELECT country_name, year, real_expenditure '\
            'FROM '\
            'boost.edu_private_expenditure_by_country_year '
    df = execute_query(query)
    return df

def get_education_indicator():
    query= 'SELECT * '\
            'FROM '\
            'indicator.global_data_lab_hd_index '

    df = execute_query(query)
    return df

def get_learning_poverty_rate():
    query = 'SELECT * '\
            'FROM '\
            'indicator.learning_poverty_rate'
    df = execute_query(query)
    return df

def get_expenditure_by_country_func_econ_year():
    return execute_query("""
        SELECT * FROM boost.expenditure_by_country_func_econ_year
        ORDER BY country_name, func, econ, year
    """)
