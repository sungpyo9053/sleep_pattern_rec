from abc import ABC, abstractmethod


class AbstractModel(ABC):
    """
    Abstract model that can predict outputs for given inputs.
    It doesn't always need training step, since it can be, e.g., ensemble,
    that combines already trained models.

    In addition, abstract model provides name, possible followed by some suffix.
    """

    def __init__(self, name_suffix):
        """
        :param name_suffix: The suffix for the name
        """
        self.__name_suffix = name_suffix

    @abstractmethod
    def predict(self, in_records):
        pass

    def get_name(self):
        return f'{self._inner_get_name()}{self.__name_suffix}'

    @abstractmethod
    def _inner_get_name(self):
        pass


class AbstractFitableModel(AbstractModel):
    """
    A model derived from AbstractModel, that requires training step.
    Since we use different dataset variants, which made by exclusion of some columns
    from the overall dataset.
    """

    def __init__(self, cols_to_take, name_suffix):
        """
        :param cols_to_take: 0-based column number list containing indices of columns included into processing
        :param name_suffix: Suffix for the name
        """
        super().__init__(name_suffix)

        self._cols_to_take = cols_to_take

    def predict(self, in_records):
        return self._inner_predict(in_records[:, self._cols_to_take])

    @abstractmethod
    def _inner_predict(self, in_records):
        """
        Performs the outputs prediction from the given inputs.
        Input columns set are modified according to the given cols_to_take.
        :param in_records: The inputs used to predict the outputs
        :return: The predictions
        """
        pass

    def fit(self, in_train, out_train):
        return self._inner_fit(in_train[:, self._cols_to_take], out_train)

    @abstractmethod
    def _inner_fit(self, in_train, out_train):
        """
        Performs the training.
        Input columns set are modified according to the given cols_to_take.
        :param in_train: Train inputs
        :param out_train: Train outputs
        :return: Nothing
        """
        pass
