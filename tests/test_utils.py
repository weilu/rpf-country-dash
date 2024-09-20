import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from utils import filter_country_sort_year

class TestUtils(unittest.TestCase):

    def setUp(self):
        self.data = {
            'country_name': ['USA', 'Canada', 'USA', 'Mexico', 'USA', 'Canada'],
            'year': [1999, 2015, 2021, 2018, 2010, 2001],
            'earliest_year': [1999, 2001, 1999, 2018, 1999, 2001],
            'value': [100, 200, 300, 400, 500, 600]
        }
        self.df = pd.DataFrame(self.data)

    def test_filter_by_country(self):
        # Test filtering by country "USA" and sorting
        expected_data = {
            'country_name': ['USA', 'USA', 'USA'],
            'year': [2021, 2010, 1999],
            'earliest_year': [1999, 1999, 1999],
            'value': [300, 500, 100]
        }
        expected_df = pd.DataFrame(expected_data)
        result_df = filter_country_sort_year(self.df, 'USA', start_year=0)
        assert_frame_equal(result_df.reset_index(drop=True), expected_df)

    def test_filter_no_results(self):
        # Test filtering by country not present in the dataframe
        result_df = filter_country_sort_year(self.df, 'France')
        self.assertTrue(result_df.empty)

    def test_filter_by_country_with_start_year(self):
        # Test filtering by country "Canada"
        # only keeping rows >= start & earliest_year is updated
        expected_data = {
            'country_name': ['Canada'],
            'year': [2015],
            'earliest_year': [2015],
            'value': [200]
        }
        expected_df = pd.DataFrame(expected_data)
        result_df = filter_country_sort_year(self.df, 'Canada', start_year=2010)
        assert_frame_equal(result_df.reset_index(drop=True), expected_df)


if __name__ == '__main__':
    unittest.main()
