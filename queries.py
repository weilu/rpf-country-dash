import os
import pandas as pd
from databricks import sql

SERVER_HOSTNAME = os.getenv("SERVER_HOSTNAME")
HTTP_PATH = os.getenv("HTTP_PATH")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
PUBLIC_ONLY = os.getenv("PUBLIC_ONLY", "False").lower() in ("true", "1", "yes")


class QueryService:
    _instance = None

    @staticmethod
    def get_instance():
        if QueryService._instance is None:
            QueryService._instance = QueryService()
        return QueryService._instance

    def __init__(self):
        self.country_whitelist = None
        if PUBLIC_ONLY:
            query = """
                SELECT country_name
                FROM prd_mega.boost.data_availability
                WHERE boost_public = 'Yes'
            """
            self.country_whitelist = self.execute_query(query)["country_name"].tolist()

    def execute_query(self, query):
        """
        Executes a query and returns the result as a pandas DataFrame.
        """
        conn = sql.connect(
            server_hostname=SERVER_HOSTNAME,
            http_path=HTTP_PATH,
            access_token=ACCESS_TOKEN,
        )
        cursor = conn.cursor()
        cursor.execute(query)
        df = cursor.fetchall_arrow().to_pandas()
        cursor.close()
        conn.close()
        return df

    def fetch_data(self, query):
        df = self.execute_query(query)
        return self._apply_country_whitelist_filter(df)

    def _apply_country_whitelist_filter(self, df):
        if self.country_whitelist is not None and "country_name" in df.columns:
            return df[df["country_name"].isin(self.country_whitelist)]
        return df

    def get_expenditure_w_poverty_by_country_year(self):
        query = """
            SELECT *
            FROM prd_mega.boost.pov_expenditure_by_country_year
        """
        df = self.fetch_data(query)
        df.loc[:, "decentralized_expenditure"] = df["decentralized_expenditure"].fillna(
            0
        )
        return df

    def get_edu_private_expenditure(self):
        query = """
            SELECT country_name, year, real_expenditure
            FROM prd_mega.boost.edu_private_expenditure_by_country_year
        """
        return self.fetch_data(query)

    def get_hd_index(self, countries):
        country_list = "', '".join(countries)
        query = f"""
            SELECT * FROM prd_mega.indicator.global_data_lab_hd_index
            WHERE country_name IN ('{country_list}')
            ORDER BY country_name, year
        """
        return self.fetch_data(query)

    def get_learning_poverty_rate(self):
        query = """
            SELECT * FROM prd_mega.indicator.learning_poverty_rate
        """
        return self.fetch_data(query)

    def get_expenditure_by_country_func_econ_year(self):
        query = """
            SELECT * FROM prd_mega.boost.expenditure_by_country_func_econ_year
        """
        return self.fetch_data(query)

    def get_expenditure_by_country_sub_func_year(self):
        query = """
            SELECT country_name, geo0, year, func, latest_year, func_sub, expenditure, real_expenditure
            FROM prd_mega.boost.expenditure_by_country_geo0_func_sub_year
        """
        return self.fetch_data(query)

    def get_basic_country_data(self, countries):
        country_list = "', '".join(countries)
        query = f"""
            SELECT country_name, display_lon, display_lat, zoom, income_level
            FROM prd_mega.indicator.country
            WHERE country_name IN ('{country_list}')
        """
        return self.fetch_data(query)

    def get_expenditure_by_country_geo1_year(self):
        query = """
            SELECT country_name, year, adm1_name, expenditure, per_capita_expenditure
            FROM prd_mega.boost.expenditure_by_country_geo1_year
        """
        return self.fetch_data(query)

    def get_adm_boundaries(self, countries):
        country_list = "', '".join(countries)
        query = f"""
            SELECT country_name, admin1_region, boundary
            FROM prd_mega.indicator.admin1_boundaries_gold
            WHERE country_name IN ('{country_list}')
            ORDER BY country_name
        """
        return self.fetch_data(query)

    def get_subnational_poverty_index(self, countries):
        country_list = "', '".join(countries)
        query = f"""
            SELECT * FROM prd_mega.indicator.subnational_poverty_index
            WHERE country_name IN ('{country_list}')
        """
        return self.fetch_data(query)

    def get_universal_health_coverage_index(self):
        query = """
            SELECT * FROM prd_mega.indicator.universal_health_coverage_index_gho
        """
        return self.fetch_data(query)

    def get_health_private_expenditure(self):
        query = """
            SELECT country_name, year, real_expenditure
            FROM prd_mega.boost.health_private_expenditure_by_country_year
        """
        return self.fetch_data(query)

    def expenditure_and_outcome_by_country_geo1_func_year(self):
        query = """
            SELECT * FROM prd_mega.boost.expenditure_and_outcome_by_country_geo1_func_year
        """
        return self.fetch_data(query)

    def get_pefa(self, countries):
        country_list = "', '".join(countries)
        query = f"""
            SELECT * FROM prd_mega.indicator.pefa_by_pillar
            WHERE country_name IN ('{country_list}')
            ORDER BY country_name, year
        """
        return self.fetch_data(query)
