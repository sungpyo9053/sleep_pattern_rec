from collections import namedtuple

import numpy as np

ALL_METRICS = ['rmse', 'mae', 'aoc_n']
EvaluationResults = namedtuple('EvaluationResults', 'rmse mae aoc_n')


def get_evaluation_results(expected, actual):
    return EvaluationResults(
        rmse=records_rmse(expected, actual),
        mae=records_mae(expected, actual),
        aoc_n=records_aoc_n(expected, actual)
    )


def get_evaluation_results_for_user(expected, actual, user_ids, user_id):
    idx = user_ids == user_id
    u_expected = expected[idx]
    u_actual = actual[idx]

    return get_evaluation_results(u_expected, u_actual)


def records_rmse(expected, actual):
    return np.sqrt(np.mean((expected - actual) ** 2))


def records_mae(expected, actual):
    return np.mean(np.abs(expected - actual))


def records_rroc(expected, actual):
    errors = actual - expected  # important, function is asymmetric
    errors = np.sort(errors)

    val_count = len(errors)
    res_size = val_count + 2

    rroc_x = np.zeros(res_size)
    rroc_y = np.zeros(res_size)

    for i in range(0, val_count):
        shift = -errors[i]
        tmp = errors + shift
        rroc_x[i + 1] = np.sum(tmp[tmp > 0])
        rroc_y[i + 1] = np.sum(tmp[tmp <= 0])

    rroc_x[res_size - 1] = np.PINF
    rroc_y[0] = np.NINF

    return rroc_x, rroc_y


def records_aoc(expected, actual):
    rroc_x, rroc_y = records_rroc(expected, actual)
    res = 0

    for i in range(1, len(rroc_x) - 2):
        res += 0.5 * (rroc_y[i + 1] + rroc_y[i]) * (rroc_x[i + 1] - rroc_x[i])

    return res


def records_aoc_n(expected, actual):
    return records_aoc(expected=expected, actual=actual) / len(expected)


def eval_by_users(expected, actual, user_ids, eval_f):
    """
    Generates the map contains evaluation results for each user.
    :param expected: Expected values vector
    :param actual: Actual values vector
    :param user_ids: The vector of user ids that corresponds given expected and actual vectors
    :param eval_f: The evaluation function
    :return: The map User ID -> Evaluation metric value
    """
    users = np.unique(user_ids)
    res = {}

    for u in users:
        idx = user_ids == u
        u_expected = expected[idx]
        u_actual = actual[idx]

        res[int(u)] = eval_f(expected=u_expected, actual=u_actual)

    return res
