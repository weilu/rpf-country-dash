import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from queries import QueryService, PUBLIC_ONLY

class TestQueryService(unittest.TestCase):

    def setUp(self):
        self.patcher = patch.object(QueryService, "execute_query")
        self.mock_execute_query = self.patcher.start()
        self.mock_table_df = pd.DataFrame({
            'country_name': ['Country1', 'Country2'],
            'year': [2020, 2021],
            'value': [100, 200]
        })
        self.mock_execute_query.return_value = self.mock_table_df

        self.query_service = QueryService.get_instance()

    def tearDown(self):
        # Reset the singleton instance after each test to avoid test interference
        QueryService._instance = None

        self.patcher.stop()


    def test_singleton_instance(self):
        instance1 = QueryService.get_instance()
        instance2 = QueryService.get_instance()
        self.assertIs(instance1, instance2)

    def test_init_country_whitelist_conditions_on_public_only(self):
        with patch("queries.PUBLIC_ONLY", False):
            service = QueryService()
            self.assertIsNone(service.country_whitelist)

        with patch("queries.PUBLIC_ONLY", True):
            service = QueryService()
            self.assertIsNotNone(service.country_whitelist)

    @patch("queries.PUBLIC_ONLY", False)
    def test_fetch_data_no_filter(self):
        df = self.query_service.fetch_data("SELECT * FROM test_table")
        pd.testing.assert_frame_equal(df, self.mock_table_df)

    @patch("queries.PUBLIC_ONLY", True)
    def test_fetch_data_applies_country_whitelist(self):
        # Mock country_whitelist
        self.query_service.country_whitelist = ["Country1"]

        df = self.query_service.fetch_data("SELECT * FROM test_table")
        expected_df = self.mock_table_df[self.mock_table_df['country_name'] == 'Country1']
        pd.testing.assert_frame_equal(df, expected_df)

    @patch("queries.PUBLIC_ONLY", True)
    def test_get_expenditure_w_poverty_by_country_year(self):
        self.query_service.country_whitelist = ["Country1"]
        self.mock_table_df['decentralized_expenditure'] = None

        df = self.query_service.get_expenditure_w_poverty_by_country_year()

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["country_name"], "Country1")
        self.assertEqual(df.iloc[0]["decentralized_expenditure"], 0)

if __name__ == "__main__":
    unittest.main()

