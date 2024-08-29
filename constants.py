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


NARRATIVE_ERROR_TEMPLATES = {
    "DATA_UNAVAILABLE": "Data not available for this period.",
    "DATA_UNAVAILABLE_DATASET_NAME": "{dataset_name} data not available for this period.",
    "GENERIC_ERROR": "An error occurred while processing the data.",
}
