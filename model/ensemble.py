import numpy as np
from abc import ABCMeta, abstractmethod
from recordclass import recordclass

from model.abstract import AbstractModel
from sd_util.evaluator import ALL_METRICS, records_rmse, eval_by_users
from sd_util.reader import USER_ID_COL


class AbstractEnsembleModel(AbstractModel, metaclass=ABCMeta):
    """
    Abstract class for building weighted ensemble models.
    """

    def __init__(self, models, name_suffix='', user_id_col=USER_ID_COL):
        """
        :param models: The models to combine
        :param name_suffix: The suffix for the name
        :param user_id_col: 0-based index of user_id column in the dataset
        """
        super().__init__(name_suffix)

        self.models = models
        self._user_id_col = user_id_col
        self._weights = None

        self._calc_weights()

    @abstractmethod
    def _calc_weights(self):
        """
        Fills the self._weights field, that must be the following map:
        user_id -> { model_name -> weight }
        """
        pass

    def predict(self, in_records):
        """
        Computes a prediction for given inputs using the self._weights field.
        For each user, the output is calculated as sum of weighted outputs of self.models.
        :param in_records: The inputs used to predict the outputs
        :return: The predictions
        """
        res = np.zeros(shape=len(in_records))

        for model in self.models:
            prediction = model.predict(in_records)
            for user_id, model_weights in self._weights.items():
                idx = in_records[:, self._user_id_col] == user_id
                prediction[idx] *= model_weights[model.get_name()]

            res += prediction

        return res


# Simple switching ensemble model

class SwitchingEnsembleModel(AbstractEnsembleModel):
    """
    The ensemble model based on a simple switching strategy.
    For each user, the model is used that showed the best results (according to the given metric) in the previous test.
    """

    def __init__(self, models, eval_results, name_suffix='', metric_in_use='rmse', user_id_col=USER_ID_COL):
        """
        :param models: The models to combine
        :param eval_results: The map {model_name -> {user_id -> {metric_name -> metric_value}}
        :param name_suffix: The suffix for the name
        :param metric_in_use: The metric that used to choose the best model for user
        :param user_id_col: 0-based index of user_id column in the dataset
        """
        self.eval_results = eval_results
        self.metric_in_use = metric_in_use
        self._metric_in_use_idx = ALL_METRICS.index(metric_in_use)

        super().__init__(models=models, name_suffix=name_suffix, user_id_col=user_id_col)

    def _calc_weights(self):
        best_eval_results = {}  # user_id -> best metric value
        best_eval_models = {}  # user_id -> best model
        for model_name, all_eval_results in self.eval_results.items():
            for user_id, eval_results in [(k, v) for k, v in all_eval_results.items() if k != 'Total']:
                metric_val = eval_results[self._metric_in_use_idx]
                if user_id not in best_eval_results or best_eval_results[user_id] > metric_val:
                    best_eval_models[user_id] = model_name
                    best_eval_results[user_id] = metric_val

        self._weights = dict(
            (k, dict(
                (model_name, 1 if best_eval_models[k] == model_name else 0)
                for model_name in self.eval_results.keys()
            ))
            for k in best_eval_results.keys()
        )

    def _inner_get_name(self):
        return 'ES'


# Ensemble model with evolutionary weight fitting

ScoredIndividual = recordclass('ScoredIndividual', 'individual eval_val by_user_eval_val')


class WeightFittingEnsembleModel(AbstractEnsembleModel):
    """
    The ensemble model based on the weighted hybridization strategy with the evolutionary
    weight-fitting algorithm.
    """

    def __init__(self, models, in_test, out_test, name_suffix='',
                 eval_f=records_rmse, gen_until_stop=50, population_size=50, mutation_probability=0.7,
                 print_learning_debug=True, user_id_col=USER_ID_COL):
        """
        :param models: The models to combine
        :param in_test: The inputs used to fit weights
        :param out_test: The outputs used to fit weights
        :param name_suffix: The suffix for the name
        :param eval_f: The evaluation function minimized when fitting weights
        :param gen_until_stop: When model shows the same result for gen_until_stop sequential generations, the
        genetic algorithm stops
        :param population_size: The size of the population for the genetic algorithm
        :param mutation_probability: The probability of the mutation for the genetic algorithm
        :param print_learning_debug: If true, evaluation metrics values will be showed for each generation while
        fitting weights
        :param user_id_col: 0-based index of user_id column in the dataset
        """
        self.in_test = in_test
        self.out_test = out_test
        self.__eval_f = eval_f
        self.__user_ids_in_test = self.in_test[:, user_id_col]
        self.__user_ids = sorted(np.unique(self.__user_ids_in_test, axis=0).tolist())
        self.__population_size = population_size
        self.__mutation_probability = mutation_probability
        self.__gen_until_stop = gen_until_stop
        self.__print_learning_debug = print_learning_debug

        super().__init__(models=models, name_suffix=name_suffix, user_id_col=user_id_col)

    def _calc_weights(self):
        # Initializing the population
        population = [self.__make_individual() for i in range(self.__population_size)]

        same_gen_count = 0
        prev_gen_score = np.PINF
        gen = 1
        population_half_size = self.__population_size // 2
        half_indices = [i for i in range(population_half_size)]

        while same_gen_count < self.__gen_until_stop:
            # Evaluating
            # If we evaluate 2nd, 3rd, ... generation, the evaluation metric values for the first half
            # of the population is already evaluated
            evaluate_from = 0 if gen == 1 else population_half_size

            for individual in population[evaluate_from:]:
                # Evaluating the individual
                self._weights = self.__transform_individual_to_weights(individual)
                prediction = self.predict(self.in_test)

                # Getting metric values
                individual.eval_val = self.__eval_f(expected=self.out_test, actual=prediction)
                individual.by_user_eval_val = eval_by_users(
                    expected=self.out_test, actual=prediction, user_ids=self.__user_ids_in_test, eval_f=self.__eval_f)

            # Selecting best parents (parent count equals to the half of population size)
            population.sort(key=lambda x: x.eval_val)
            population = population[0:population_half_size]

            # Performing crossover:
            # Probability is higher if eval_val is lower
            p = np.array([i.eval_val for i in population])
            p = np.flip(p / p.sum())
            taken_pairs = set()
            for i in range(population_half_size):
                while True:
                    first_idx, second_idx = np.random.choice(half_indices, 2, p=p)
                    if first_idx != second_idx and \
                            (first_idx, second_idx) not in taken_pairs and \
                            (second_idx, first_idx) not in taken_pairs:
                        break

                taken_pairs.add((first_idx, second_idx))
                population.append(self.__crossover(population[first_idx], population[second_idx]))

            # Performing mutations with given probability
            for i in range(population_half_size, self.__population_size):
                if np.random.randint(1, 100) < 100 * self.__mutation_probability:
                    self.__mutate(population[i])

            # Calculating stop conditions
            if population[0].eval_val == prev_gen_score:
                same_gen_count += 1
            else:
                same_gen_count = 0
                prev_gen_score = population[0].eval_val

            # Printing debug values if needed
            if self.__print_learning_debug:
                print(f'{gen}: {population[0].eval_val}')

            gen += 1

        self._weights = self.__transform_individual_to_weights(population[0])

    def __make_individual(self):
        """
        The individual is considered as matrix sized (user_count, model_count) that contains
        weights of particular models for particular users.
        The weights initialized randomly, and their sum for each user equals 1.

        :return: New individual
        """
        user_count = len(self.__user_ids)
        model_count = len(self.models)

        individual = np.zeros(shape=(user_count, model_count))
        for i in range(user_count):
            row = np.random.random(model_count)
            row /= row.sum()
            individual[i] = row

        return ScoredIndividual(individual=individual, eval_val=None, by_user_eval_val=None)

    def __transform_individual_to_weights(self, individual):
        """
        Transforms given individual (i.e. matrix of weights) to the map user_id -> { model_name -> weight }
        that corresponds self._weights format.
        :param individual: The individual (i.e. matrix of weights)
        :return: The map user_id -> { model_name -> weight }
        """
        return dict(
            (self.__user_ids[i], dict(
                (self.models[j].get_name(), individual.individual[i, j])
                for j in range(len(self.models))
            ))
            for i in range(len(self.__user_ids))
        )

    def __crossover(self, first_parent, second_parent):
        """
        Crosses two parents and returns their child. For each user, the row in the child matrix
        equals the row from the parent showed the best result for this user.
        :param first_parent: The parent individual (i.e. matrix of weights)
        :param second_parent: The parent individual (i.e. matrix of weights)
        :return: The child individual (i.e. matrix of weights)
        """
        individual = np.zeros(shape=first_parent.individual.shape)
        for i in range(len(self.__user_ids)):
            first_parent_metric = first_parent.by_user_eval_val[self.__user_ids[i]]
            second_parent_metric = first_parent.by_user_eval_val[self.__user_ids[i]]

            if first_parent_metric < second_parent_metric:
                individual[i] = first_parent.individual[i]
            else:
                individual[i] = second_parent.individual[i]

        return ScoredIndividual(individual=individual, eval_val=None, by_user_eval_val=None)

    def __mutate(self, individual):
        """
        Mutates the given individual. Mutation is performed in the following way.
        Two columns are randomly selected, and one of two mutations is performed with the same probability:
        1. Swapping these columns
        2. Adding 0.05 to the first column and subtracting 0.05 from the second column

        :param individual: The individual (i.e. matrix of weights)
        :return: Nothing
        """
        model_count = len(self.models)
        col1 = np.random.randint(0, model_count - 1)
        col2 = col1
        while col2 == col1:
            col2 = np.random.randint(0, model_count - 1)

        if np.random.randint(1, 100) > 50:
            individual.individual[:, [col1, col2]] = individual.individual[:, [col2, col1]]
        else:
            individual.individual[:, col1] += 0.05
            individual.individual[:, col2] -= 0.05

    def _inner_get_name(self):
        return 'EWF'
