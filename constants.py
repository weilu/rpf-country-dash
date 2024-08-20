from enum import Enum


START_YEAR = 2010

CORRELATION_THRESHOLDS = {
    "0": "no correlation",
    "0.1": "negligible correlation",
    "0.4": "weak correlation",
    "0.7": "moderate correlation",
    "0.9": "strong correlation",
    "1": "very strong correlation",
}

TREND_THRESHOLDS = 0.4


class NARRATIVE_TEMPLATE(Enum):

    DATA_UNAVAILABLE = "Data not available for this period."
    GENERIC_ERROR = "An error occurred while processing the data."
