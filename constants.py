from enum import Enum


START_YEAR = 2010

CORRELATION_THRESHOLDS = {
    "0": "no",
    "0.1": "no",
    "0.3": "weak",
    "0.7": "moderate",
    "0.9": "strong",
    "1": "very strong",
}

TREND_THRESHOLDS = 0.4


NARRATIVE_ERROR_TEMPLATES = {
    "DATA_UNAVAILABLE": "Data not available for this period.",
    "DATA_UNAVAILABLE_DATASET_NAME": "{dataset_name} data not available for this period.",
    "GENERIC_ERROR": "An error occurred while processing the data.",
}
