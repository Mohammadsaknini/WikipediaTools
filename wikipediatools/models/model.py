from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    An abstract class used to represent a model.
    """

    @abstractmethod
    def predict(self):
        """
        Predicts the quality of a Wikipedia page.
        """
        pass

    @abstractmethod
    def get_params(self):
        """
        Returns the parameters of the model.
        """
        pass

    @abstractmethod
    def feature_importance(self, plot_type="bar"):
        """
        Returns the feature importance of the model.
        """
        pass

    @abstractmethod
    def classification_metrics(self):
        """
        Returns the classification metrics of the model.
        """
        pass
