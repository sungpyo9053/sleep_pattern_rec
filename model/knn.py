from sklearn.neighbors import KNeighborsRegressor

from model.abstract import AbstractFitableModel
from sd_util.evaluator import records_rmse
from sd_util.reader import read_rate_records, split_rate_records, WITHOUT_CONTEXT_COLUMNS, USER_ID_COL


class KNNModel(AbstractFitableModel):
    """
    The model implements collaborative filtering using k-NN algorithm.
    """

    def __init__(self, cols_to_take, name_suffix='', neighbour_count=4):
        super().__init__(cols_to_take, name_suffix)
        self.__knn_model = KNeighborsRegressor(n_neighbors=neighbour_count,
                                               weights='distance',
                                               algorithm='kd_tree')

    def _inner_fit(self, in_train, out_train):
        self.__knn_model.fit(in_train, out_train)

    def _inner_predict(self, in_records):
        return self.__knn_model.predict(in_records)

    def _inner_get_name(self):
        return 'k-NN'


def main():
    model = KNNModel(cols_to_take=WITHOUT_CONTEXT_COLUMNS)

    in_data, out_data = read_rate_records()
    in_train, in_test, out_train, out_test = split_rate_records(in_data, out_data, stratify=in_data[:, USER_ID_COL])

    model.fit(in_train, out_train)
    out_predicted = model.predict(in_test)

    print(records_rmse(out_test, out_predicted))


if __name__ == '__main__':
    main()
