import numpy as np
import pandas as pd
from itertools import chain

from model.knn import KNNModel
from model.mlp import MLPModel
from model.rf import RandomForestModel
from model.svd import SVDModel
from model.ensemble import SwitchingEnsembleModel, WeightFittingEnsembleModel
from sd_util.evaluator import get_evaluation_results, get_evaluation_results_for_user, ALL_METRICS
from sd_util.reader import read_rate_records, split_rate_records, USER_ID_COL, WITHOUT_CONTEXT_COLUMNS, \
    WITH_WEEKEND_CONTEXT_COLUMNS, WITH_SLEEP_NOTES_CONTEXT_COLUMNS, WITH_WHOLE_CONTEXT_COLUMNS

# Suffixes for model names:

WITHOUT_CONTEXT_SUFFIX = ' (w/o context)'
WITH_WEEKEND_CONTEXT_SUFFIX = ' (w/ weekends)'
WITH_SLEEP_NOTES_CONTEXT_SUFFIX = ' (w/ sleep notes)'
WITH_WHOLE_CONTEXT_SUFFIX = ' (w/ whole context)'


# Helper functions:

def test_models(models, in_train, in_test, out_train, out_test, fit=True):
    """
    Trains the given models if needed and evaluates them, returning the evaluation results.

    :param models: The models to train and evaluate
    :param in_train: Train part inputs
    :param in_test: Test part inputs
    :param out_train: Train part outputs
    :param out_test: Test part outputs
    :param fit: If true then models will be trained
    :return: The map {Model name -> {User ID -> EvaluationResults}}
    """
    user_ids_in_test = in_test[:, USER_ID_COL]
    res = {}

    for m in models:
        if fit:
            m.fit(in_train, out_train)
        predicted = m.predict(in_test)

        m_res = {'Total': get_evaluation_results(expected=out_test, actual=predicted)}
        for user_id in user_ids_un:
            m_res[user_id] = get_evaluation_results_for_user(expected=out_test, actual=predicted,
                                                             user_ids=user_ids_in_test, user_id=user_id)
            res[m.get_name()] = m_res

    return res


def create_best_results_df(evaluation_results):
    """
    Converts the result of test_models function to the DataFrame that represents the best results
    in the simple table.

    :param evaluation_results: The map {Model name -> {User ID -> EvaluationResults}}
    :return: DataFrame contains one row per user and total row, and one columns per evaluation metric
    """
    index = user_ids_un + ['Total']
    columns = ALL_METRICS

    res_eval_values = pd.DataFrame(np.PINF, index=index, columns=columns)
    res_best_models = pd.DataFrame('', index=index, columns=columns)

    for model_name, all_eval_results in evaluation_results.items():
        for idx, eval_results in all_eval_results.items():
            for i in range(len(ALL_METRICS)):
                if eval_results[i] < res_eval_values[ALL_METRICS[i]][idx]:
                    res_eval_values[ALL_METRICS[i]][idx] = eval_results[i]
                    res_best_models[ALL_METRICS[i]][idx] = model_name
                elif eval_results[i] == res_eval_values[ALL_METRICS[i]][idx]:
                    res_best_models[ALL_METRICS[i]][idx] += f', {model_name}'

    res = pd.DataFrame('', index=index, columns=columns)
    for i in index:
        for col in columns:
            res[col][i] = f'{res_best_models[col][i]} ({res_eval_values[col][i]:.3f})'

    return res


def create_all_results_df(evaluation_results):
    """
        Converts the result of test_models function to the DataFrame that represents all the results
        in the simple table.

        :param evaluation_results: The map {Model name -> {User ID -> EvaluationResults}}
        :return: DataFrame contains one row per user and total row, and columns with evaluation metric values
        for all the models
        """
    index = user_ids_un + ['Total']
    columns = list(chain(*[
        [f'{model_name} ({m})' for m in ALL_METRICS] for model_name in sorted(evaluation_results.keys())
    ]))

    res = pd.DataFrame(np.nan, index=index, columns=columns)
    for model_name, all_eval_results in evaluation_results.items():
        for idx, eval_results in all_eval_results.items():
            for i in range(len(ALL_METRICS)):
                column = f'{model_name} ({ALL_METRICS[i]})'
                res[column][idx] = eval_results[i]

    return res


def concat_dicts(*dicts):
    return dict(chain.from_iterable(d.items() for d in dicts))


def create_and_test_regular_models(cols_to_take, name_suffix):
    """
    Creates 4 models (SVD, k-NN, RF and MLP) which takes cue from given columns.
    Next, these models are trained and evaluated.

    :param cols_to_take: 0-based column number list containing indices of columns included into processing
    :param name_suffix: The suffix for models' names
    :return: List of trained models and the map {Model name -> {User ID -> EvaluationResults }}
    """

    models = [
        SVDModel(all_data=in_data, cols_to_take=cols_to_take, name_suffix=name_suffix),
        KNNModel(cols_to_take=cols_to_take, name_suffix=name_suffix),
        RandomForestModel(cols_to_take=cols_to_take, name_suffix=name_suffix),
        MLPModel(cols_to_take=cols_to_take, name_suffix=name_suffix)
    ]

    eval_res = test_models(
        models=models,
        in_train=in_for_regular_train, in_test=in_for_regular_test,
        out_train=out_for_regular_train, out_test=out_for_regular_test
    )

    return models, eval_res


########################################################################################################################
#                                                    The main part                                                     #
########################################################################################################################

# Reading and splitting data. As it is mentioned in the paper:
# 1. Regular models are trained on C and evaluated on D
# 2. Ensemble model is evaluated on B

in_data, out_data = read_rate_records('data/sleepdata_classified_2.csv')

in_, in_for_ensemble_test, out_, out_for_ensemble_test = \
    split_rate_records(in_data, out_data, test_part=0.2, stratify=in_data[:, USER_ID_COL])

in_for_regular_train, in_for_regular_test, out_for_regular_train, out_for_regular_test = \
    split_rate_records(in_, out_, test_part=0.2, stratify=in_[:, USER_ID_COL])

# Getting user ids list for report purposes

user_ids_un = [int(u) for u in sorted(np.unique(in_data[:, USER_ID_COL], axis=0).tolist())]

# Creating regular models

without_context_models, without_context_eval_res = create_and_test_regular_models(
    WITHOUT_CONTEXT_COLUMNS, WITHOUT_CONTEXT_SUFFIX)

with_weekend_context_models, with_weekend_context_eval_res = create_and_test_regular_models(
    WITH_WEEKEND_CONTEXT_COLUMNS, WITH_WEEKEND_CONTEXT_SUFFIX)

with_sleep_notes_context_models, with_sleep_notes_context_eval_res = create_and_test_regular_models(
    WITH_SLEEP_NOTES_CONTEXT_COLUMNS, WITH_SLEEP_NOTES_CONTEXT_SUFFIX)

with_whole_context_models, with_whole_context_eval_res = create_and_test_regular_models(
    WITH_WHOLE_CONTEXT_COLUMNS, WITH_WHOLE_CONTEXT_SUFFIX)

# Finding best models before hybridization

overall_eval_res_before_hybridization = concat_dicts(
    without_context_eval_res,
    with_weekend_context_eval_res,
    with_sleep_notes_context_eval_res,
    with_whole_context_eval_res
)
best_results_before_hybridization = create_best_results_df(overall_eval_res_before_hybridization)
all_results_before_hybridization = create_all_results_df(overall_eval_res_before_hybridization)
print(best_results_before_hybridization)

# Performing hybridization

all_models = without_context_models + \
             with_weekend_context_models + \
             with_sleep_notes_context_models + \
             with_whole_context_models

all_contextual_models = with_weekend_context_models + \
                        with_sleep_notes_context_models + \
                        with_whole_context_models

contextual_model_eval_res = concat_dicts(
    with_sleep_notes_context_eval_res,
    with_weekend_context_eval_res,
    with_whole_context_eval_res
)

ensemble_models = [
    SwitchingEnsembleModel(
        name_suffix=' (all)',
        models=all_models,
        eval_results=overall_eval_res_before_hybridization
    ),
    SwitchingEnsembleModel(
        name_suffix=' (w/ some context)',
        models=all_contextual_models,
        eval_results=contextual_model_eval_res
    ),
    SwitchingEnsembleModel(
        name_suffix=WITH_WHOLE_CONTEXT_SUFFIX,
        models=with_whole_context_models,
        eval_results=with_whole_context_eval_res
    ),
    SwitchingEnsembleModel(
        name_suffix=WITH_SLEEP_NOTES_CONTEXT_SUFFIX,
        models=with_sleep_notes_context_models,
        eval_results=with_sleep_notes_context_eval_res
    ),
    SwitchingEnsembleModel(
        name_suffix=WITH_WEEKEND_CONTEXT_SUFFIX,
        models=with_weekend_context_models,
        eval_results=with_weekend_context_eval_res
    ),
    SwitchingEnsembleModel(
        name_suffix=WITHOUT_CONTEXT_SUFFIX,
        models=without_context_models,
        eval_results=without_context_eval_res
    ),
    WeightFittingEnsembleModel(
        name_suffix=' (all)',
        models=all_models,
        in_test=in_for_regular_test, out_test=out_for_regular_test
    ),
    WeightFittingEnsembleModel(
        name_suffix=' (w/ some context)',
        models=all_contextual_models,
        in_test=in_for_regular_test, out_test=out_for_regular_test
    ),
    WeightFittingEnsembleModel(
        name_suffix=WITH_WHOLE_CONTEXT_SUFFIX,
        models=with_whole_context_models,
        in_test=in_for_regular_test, out_test=out_for_regular_test
    ),
    WeightFittingEnsembleModel(
        name_suffix=WITH_SLEEP_NOTES_CONTEXT_SUFFIX,
        models=with_sleep_notes_context_models,
        in_test=in_for_regular_test, out_test=out_for_regular_test
    ),
    WeightFittingEnsembleModel(
        name_suffix=WITH_WEEKEND_CONTEXT_SUFFIX,
        models=with_weekend_context_models,
        in_test=in_for_regular_test, out_test=out_for_regular_test
    ),
    WeightFittingEnsembleModel(
        name_suffix=WITHOUT_CONTEXT_SUFFIX,
        models=without_context_models,
        in_test=in_for_regular_test, out_test=out_for_regular_test
    )
]

overall_res_after_hybridization = test_models(
    models=all_models + ensemble_models, fit=False,
    in_train=None, in_test=in_for_ensemble_test,
    out_train=None, out_test=out_for_ensemble_test
)

# Finding best models after hybridization

best_results_after_hybridization = create_best_results_df(overall_res_after_hybridization)
all_results_after_hybridization = create_all_results_df(overall_res_after_hybridization)
print(best_results_after_hybridization)

# Saving results to Excel

best_results_before_hybridization.to_excel('best_results_before_hybridization.xlsx')
all_results_before_hybridization.to_excel('all_results_before_hybridization.xlsx')
best_results_after_hybridization.to_excel('best_results_after_hybridization.xlsx')
all_results_after_hybridization.to_excel('all_results_after_hybridization.xlsx')
