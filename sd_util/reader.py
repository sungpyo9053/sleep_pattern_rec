import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Column indices constants:

USER_ID_COL = 0
LAY_TIME_COL = 1
TOTAL_COL = 2
IS_WEEKEND_COL = 3
DRANK_COFFEE_COL = 4
DRANK_TEA_COL = 5
WORKED_OUT_COL = 6
STRESSFUL_DAY_COL = 7
ATE_LATE_COL = 8
RATE_COL = 9

# The column indices lists which represent different variants of context usage:

WITHOUT_CONTEXT_COLUMNS = [USER_ID_COL, LAY_TIME_COL, TOTAL_COL]
WITH_WEEKEND_CONTEXT_COLUMNS = [USER_ID_COL, LAY_TIME_COL, TOTAL_COL, IS_WEEKEND_COL]
WITH_SLEEP_NOTES_CONTEXT_COLUMNS = [USER_ID_COL, LAY_TIME_COL, TOTAL_COL, DRANK_COFFEE_COL, DRANK_TEA_COL,
                                    WORKED_OUT_COL, STRESSFUL_DAY_COL, ATE_LATE_COL]
WITH_WHOLE_CONTEXT_COLUMNS = [USER_ID_COL, LAY_TIME_COL, TOTAL_COL, DRANK_COFFEE_COL, DRANK_TEA_COL,
                              WORKED_OUT_COL, STRESSFUL_DAY_COL, ATE_LATE_COL, IS_WEEKEND_COL]


def read_rate_df(file_name='../data/sleepdata_classified_2.csv'):
    data = pd.read_csv(file_name)
    data = data[[
        'user_id', 'lay_time', 'total',
        'is_weekend',
        'drank_coffee', 'drank_tea', 'worked_out', 'stressful_day', 'ate_late',
        'rate'
    ]]

    lay_time_mapping = {}
    for i in range(0, 24):
        lay_time_mapping[f'{i}_{i + 1}'] = i

    data = data.replace({'lay_time': lay_time_mapping})

    return data


def read_rate_records(file_name='../data/sleepdata_classified_2.csv'):
    data = read_rate_df(file_name)
    data = data.values

    return data[:, USER_ID_COL:RATE_COL], data[:, RATE_COL]


def create_user_id_mapping(data):
    """
    :param data: Dataset (as numpy matrix)
    :return: User count and function that maps record to User ID
    """
    user_ids = np.unique(data[:, USER_ID_COL], axis=0)

    def map_user_id(record):
        return int(record[USER_ID_COL]) - 1

    return len(user_ids), map_user_id


def create_item_id_mapping(item_cols, data):
    """
    :param data: Dataset (as numpy matrix)
    :return: Item count and function that maps record to Item ID
    """
    # Data contain all columns, so we use item_cols to create item_id_mapping keys
    item_data = data[:, item_cols]
    item_multi_ids = np.unique(item_data, axis=0)

    item_id_mapping = dict((item_multi_ids[i].tobytes(), i) for i in range(item_multi_ids.shape[0]))

    def map_item_id(record):
        # When record is passed, it contains only user_id + item_cols, so we only cutting user_id
        return item_id_mapping[record[USER_ID_COL + 1:].tobytes()]

    return len(item_multi_ids), map_item_id


def split_rate_records(in_data, out_data, test_part=0.2, stratify=None):
    # Depending on the random state value, final result may slightly vary, but the overall picture doesn't change
    return train_test_split(in_data, out_data, test_size=test_part, stratify=stratify)
