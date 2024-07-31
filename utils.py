def filter_country_sort_year(df, country):
    """
    Preprocess the dataframe to filter by country and sort by year
    :param df: DataFrame
    :param country: str
    :return: DataFrame
    """
    df = df.loc[df["country_name"] == country]
    df = df.sort_values(["year"])
    return df
