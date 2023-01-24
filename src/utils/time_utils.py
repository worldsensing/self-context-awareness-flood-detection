from datetime import datetime

HOURS_TO_SECONDS = 3600
FORMAT = '%Y/%m/%d %I:%M:%S %p %z'


def translate_timestamp(t):
    return datetime.strptime(t, FORMAT)
