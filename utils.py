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


import math

millnames = ["", " K", " M", " B", " T"]


def millify(n):
    n = float(n)
    millidx = max(
        0,
        min(
            len(millnames) - 1, int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))
        ),
    )

    return "{:.2f}{}".format(n / 10 ** (3 * millidx), millnames[millidx])


def handle_empty_plot(fig):
    fig.update_layout(
        annotations=[
            dict(
                text="No Matching Data Found",
                showarrow=False,
            )
        ],
    )
    return fig
