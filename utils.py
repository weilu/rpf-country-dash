from constants import (
    NARRATIVE_ERROR_TEMPLATES,
    START_YEAR,
    CORRELATION_THRESHOLDS,
    TREND_THRESHOLDS,
)
from scipy.stats import pearsonr


def filter_country_sort_year(df, country):
    """
    Preprocess the dataframe to filter by country and sort by year
    :param df: DataFrame
    :param country: str
    :return: DataFrame
    """
    df = df.loc[df["country_name"] == country]

    df = df[df.year >= START_YEAR]
    df = df.sort_values(["year"], ascending=False)
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


def handle_empty_plot(fig, text=None):
    if not text:
        text = "No Matching Data Found"
    fig.update_layout(
        annotations=[
            dict(
                text=text,
                showarrow=False,
                x=None,
                y=None,
            )
        ],
    )
    return fig


def get_percentage_change_text(percent):
    if percent > 0:
        return f"increased by {percent:.2f}%"
    return f"decreased by {-1 * percent:.2f}%"


def calculate_PCC(df, x_col, y_col):
    """
    Calculate the Pearson Correlation Coefficient between two columns
    :param df: DataFrame
    :param x_col: str
    :param y_col: str
    :return: float
    """
    df = df[[x_col, y_col]].dropna()
    return pearsonr(df[x_col], df[y_col])[0]


def get_correlation_text(df, x_col, y_col):
    """
    Get the correlation text based on the PCC value
    :param df: DataFrame
    :param x_col: {"col_name": str, "display": str} // col_name is the column name in the DataFrame and display is the name to be displayed in the text
    :param y_col: {"col_name": str, "display": str} // col_name is the column name in the DataFrame and display is the name to be displayed in the text
    :return: str
    """

    pcc = calculate_PCC(df, x_col["col_name"], y_col["col_name"])
    if pcc > 0:
        direction = "positive "
        association = "higher"
    else:
        direction = "inverse"
        association = "lower"
    abs_pcc = abs(pcc)

    for threshold, pcc_text in CORRELATION_THRESHOLDS.items():
        if abs_pcc < float(threshold):
            intensity = pcc_text
            break

    x_display_name = x_col["display"]
    y_display_name = y_col["display"]

    if intensity == "no":
        return f"the correlation between {y_display_name} and {x_display_name} is {pcc:.2f},\
                indicating there is no linear relationship."

    text = f"the correlation between {y_display_name} and {x_display_name} is {pcc:.2f},\
                indicating a {intensity} {direction} relationship. Higher {y_display_name} is generally associated with {association} {x_display_name}."
    return text


def detect_trend(df, x_col):
    """
    Detect the trend of the data based on the PCC value
    :param df: DataFrame
    :param x_col: str
    :return: str
    """
    pcc = calculate_PCC(df, x_col, "year")
    abs_pcc = abs(pcc)
    if abs_pcc > TREND_THRESHOLDS:
        if pcc > 0:
            return "an increasing trend"
        else:
            return "a decreasing trend"

    return ""


def generate_error_prompt(template_key, **kwargs):
    """
    Generate a prompt message based on the template and the keyword arguments
    :param template: str
    :param kwargs: dict
    :return: str
    """
    template = NARRATIVE_ERROR_TEMPLATES[template_key]
    return template.format(**kwargs)
