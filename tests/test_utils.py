import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from utils import filter_country_sort_year, get_correlation_text, calculate_cagr

class TestUtils(unittest.TestCase):

    def setUp(self):
        self.data = {
            'country_name': ['USA', 'Canada', 'USA', 'Mexico', 'USA', 'Canada'],
            'year': [1999, 2015, 2021, 2018, 2010, 2001],
            'earliest_year': [1999, 2001, 1999, 2018, 1999, 2001],
            'value': [100, 200, 300, 400, 500, 600]
        }
        self.df = pd.DataFrame(self.data)

        self.x_col = {"col_name": "x", "display": "X Variable"}
        self.y_col = {"col_name": "y", "display": "Y Variable"}

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

    def test_positive_strong_correlation(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
        result = get_correlation_text(df, self.x_col, self.y_col)
        self.assertIn("very strong positive", result)

    def test_negative_moderate_correlation(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [-1, -2, -3]})
        result = get_correlation_text(df, self.x_col, self.y_col)
        self.assertIn("very strong inverse", result)

    def test_weak_correlation(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [1, 2, 1.5, 3.5, 1]})
        result = get_correlation_text(df, self.x_col, self.y_col)
        self.assertRegex(result, r"weak positive")

    def test_no_correlation(self):
        df = pd.DataFrame({"x": [1, 1, 1], "y": [2, 3, 4]})  # constant x
        result = get_correlation_text(df, self.x_col, self.y_col)
        self.assertIn("unknown due to limited data", result)

    def test_no_association(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [5, 5, 5, 5, 5]})  # constant y
        result = get_correlation_text(df, self.x_col, self.y_col)
        self.assertIn("unknown due to limited data", result)

    def test_not_enough_data_points(self):
        df = pd.DataFrame({"x": [1, 2], "y": [2, 4]})
        result = get_correlation_text(df, self.x_col, self.y_col)
        self.assertIn("unknown due to limited data", result)
        
    def test_cagr_invalid_years(self):
        self.assertIsNone(calculate_cagr(100, 200, None))
        self.assertIsNone(calculate_cagr(100, 200, 0))
        self.assertIsNone(calculate_cagr(100, 200, -2))

    def test_cagr_invalid_data(self):
        self.assertIsNone(calculate_cagr(None, 200, 5))
        self.assertIsNone(calculate_cagr(100, None, 5))
        self.assertIsNone(calculate_cagr(float('nan'), 200, 5))
        self.assertIsNone(calculate_cagr(0, 200, 5))
        self.assertIsNone(calculate_cagr(-100, 200, 5))

if __name__ == '__main__':
    unittest.main()
