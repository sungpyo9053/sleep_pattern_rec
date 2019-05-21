from sklearn.ensemble import RandomForestRegressor

from model.abstract import AbstractFitableModel
from sd_util.evaluator import records_rmse
from sd_util.reader import read_rate_records, split_rate_records, WITHOUT_CONTEXT_COLUMNS, USER_ID_COL


class RandomForestModel(AbstractFitableModel):
    """
    The model performs content analysis using random forest that contains given count of decision trees
    with unrestricted depth.
    """

    def __init__(self, cols_to_take, name_suffix='', n_estimators=100):
        super().__init__(cols_to_take, name_suffix)
        self.__rf_model = RandomForestRegressor(max_depth=None, n_estimators=n_estimators)

    def _inner_fit(self, in_train, out_train):
        self.__rf_model.fit(in_train, out_train)

    def _inner_predict(self, in_records):
        return self.__rf_model.predict(in_records)

    def _inner_get_name(self):
        return 'RF'


def main():
    in_data, out_data = read_rate_records()
    in_train, in_test, out_train, out_test = split_rate_records(in_data, out_data, stratify=in_data[:, USER_ID_COL])

    model = RandomForestModel(cols_to_take=WITHOUT_CONTEXT_COLUMNS)

    model.fit(in_train, out_train)
    out_predicted = model.predict(in_test)

    print(records_rmse(out_test, out_predicted))


if __name__ == '__main__':
    main()
