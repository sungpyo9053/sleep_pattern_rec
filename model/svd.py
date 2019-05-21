import numpy as np

from model.abstract import AbstractFitableModel
from sd_util.evaluator import records_rmse
from sd_util.reader import read_rate_records, split_rate_records, create_user_id_mapping, create_item_id_mapping, \
    USER_ID_COL, WITHOUT_CONTEXT_COLUMNS


class SVDModel(AbstractFitableModel):
    """
    The model implements collaborative filtering using SVD.
    The regularized squared error is minimized with SGD. Factors are modified using rules from
    https://sifter.org/simon/journal/20061211.html
    """

    def __init__(self, cols_to_take, all_data, name_suffix='', lam=0.00005, gamma=0.001, factor_count=5,
                 iter_count=1000):
        """
        :param cols_to_take: 0-based column number list containing indices of columns included into processing
        :param all_data: The whole dataset; it is needed to compute user and item IDs and isn't used in training!
        :param name_suffix: The suffix for the name
        :param lam: The regularization constant
        :param gamma: The learning rate
        :param factor_count: Count of the factors
        :param iter_count: Iteration count to fit the model
        """
        super().__init__(cols_to_take, name_suffix)

        self.lam = lam
        self.gamma = gamma
        self.factor_count = factor_count

        item_cols = [col for col in cols_to_take if col != USER_ID_COL]
        self.__user_count, self.__map_user_id = create_user_id_mapping(data=all_data)
        self.__item_count, self.__map_item_id = create_item_id_mapping(item_cols=item_cols, data=all_data)

        self.__iter_count = iter_count
        self.__u_factors = None
        self.__i_factors = None
        self.__prediction = None

    def __init_factors(self, row_count, factor_count=None, init=True):
        if factor_count is None:
            factor_count = self.factor_count

        return np.random.rand(row_count, factor_count) if init else np.zeros((row_count, factor_count))

    def __reg_se(self, rate_matrix, u_factors, i_factors):
        pred_rates = self.__svd(u_factors, i_factors)
        errors = rate_matrix - pred_rates
        errors[rate_matrix == 0] = 0

        u_snorms = np.linalg.norm(u_factors, axis=1) ** 2
        i_snorms = np.linalg.norm(i_factors, axis=1) ** 2
        res = errors ** 2 + np.array([u_snorms * self.lam]).T + np.array([i_snorms * self.lam])

        return res.sum(), errors

    def __sgd(self, u_factors, i_factors, errors):
        res_u_factors = u_factors + (errors.dot(i_factors) - self.lam * u_factors) * self.gamma
        res_i_factors = i_factors + (errors.T.dot(u_factors) - self.lam * i_factors) * self.gamma

        return res_u_factors, res_i_factors

    @staticmethod
    def __svd(u_factors, i_factors):
        return u_factors.dot(i_factors.T)

    def __get_indices(self, record):
        return self.__map_user_id(record), self.__map_item_id(record)

    def _inner_fit(self, in_train, out_train):
        u_factors = self.__init_factors(self.__user_count)
        i_factors = self.__init_factors(self.__item_count)

        train_matrix = np.zeros(shape=(self.__user_count, self.__item_count))
        for i in range(0, len(in_train)):
            user_idx, item_idx = self.__get_indices(in_train[i])
            train_matrix[user_idx][item_idx] = out_train[i]

        for i in range(0, self.__iter_count):
            _, errors = self.__reg_se(train_matrix, u_factors, i_factors)
            u_factors, i_factors = self.__sgd(u_factors, i_factors, errors)

        self.__u_factors = u_factors
        self.__i_factors = i_factors
        self.__prediction = self.__svd(self.__u_factors, self.__i_factors)

    def _inner_predict(self, in_records):
        res = []
        for i in range(0, len(in_records)):
            user_idx, item_idx = self.__get_indices(in_records[i])
            res.append(self.__prediction[user_idx][item_idx])

        return np.array(res)

    def _inner_get_name(self):
        return 'SVD'


def main():
    in_data, out_data = read_rate_records()
    in_train, in_test, out_train, out_test = split_rate_records(in_data, out_data, stratify=USER_ID_COL)

    model = SVDModel(cols_to_take=WITHOUT_CONTEXT_COLUMNS, all_data=in_data)

    model.fit(in_train, out_train)
    out_predicted = model.predict(in_test)

    print(records_rmse(out_test, out_predicted))


if __name__ == '__main__':
    main()
