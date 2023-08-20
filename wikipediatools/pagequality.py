from wikitools.data.features import FeaturesExtractor, FeatureSets
from wikitools.models.dnn import DeepNeuralNetwork
from wikitools.models.types import ModelType
from wikitools.models.model import BaseModel
from wikitools.models.lgbm import LightGBM
import mwparserfromhell as mwp
import pandas as pd
import requests


class QualityPredictor:
    """
    A class that predicts the quality of a Wikipedia page.
    """

    def __init__(self, features_set: FeatureSets = FeatureSets.TEXT_STATISTICS,
                 model_type: ModelType = ModelType.LightGBM):
        """
        Initialize the Wikipredictor.
        By default, it uses the Text Statistics features set and the LightGBM model which are currently the best combination.
        
        Args:
            features_set (FeaturesSet, optional): The features set to use. Defaults to FeaturesSet.TEXT_STATS_FEATURES.
            model_type (ModelTypes, optional): The model type to use. Defaults to ModelTypes.LightGBM.

        Raises:
            TypeError: If features_set is not an instance of FeaturesSet Enum.
            TypeError: If model_type is not an instance of ModelTypes Enum.
        """
        if isinstance(features_set, FeatureSets):
            self.features_set = features_set
        else:
            raise TypeError(
                "features_set must be an instance of FeaturesSet Enum")

        if isinstance(model_type, ModelType):
            self.model_type = model_type
        else:
            raise TypeError(
                "model_type must be an instance of ModelTypes Enum")

        self.model: BaseModel = None

    def _get_page(self, page: str | int) -> tuple[mwp.wikicode.Wikicode, int, str]:
        """
        Gets the page from Wikipedia.

        Args:
            page (str | int): The title of the page or the revision ID of the page to get.
            if the page is a title, the latest revision will be used.
        Returns:
            tuple[mwp.wikicode.Wikicode, int, str]: The page content, the revision ID and the title of the page.

        Raises:
            TypeError: If page is not an instance of str or int.

        """

        url = "https://en.wikipedia.org/w/api.php"

        headers = {"Accept-Encoding": "gzip"}
        params = {
            "action": "query",
            "prop": "revisions",
            "rvprop": "content|ids",
            "format": "json",
            "rvslots": "main",
        }

        if isinstance(page, str):
            params["titles"] = page
        elif isinstance(page, int):
            params["revids"] = page
        else:
            raise TypeError("page must be an instance of str or int")

        response = requests.get(url, params=params, headers=headers).json()

        # Extract the page ID from the response
        page_id = list(response["query"]["pages"].keys())[0]
        pages = response["query"]["pages"]

        revision = pages[page_id]["revisions"][0]
        content = revision["slots"]["main"]["*"]

        revid = revision["revid"]
        title = pages[page_id]["title"]

        return mwp.parse(content), revid, title

    def _get_features(self, page: str | int):
        """
        Extracts features from the sample dataset and saves it as a CSV file.

        Args:
            page (str | int): The title of the page or the revision ID of the page to get.
            if the page is a title, the latest revision will be used.
        
        Returns:
            pandas.DataFrame: The extracted features dataframe.
        """
        code, revid, title = self._get_page(page)
        extractor = FeaturesExtractor([self.features_set])
        data = extractor.extract_feature_sets(title, str(revid), code)
        data["Target"] = ""

        return pd.DataFrame([data])

    def _load_model(self, page: str | int) -> BaseModel:
        """
        Loads the model from the saved file.

        Args:
            page (str | int): The title of the page or the revision ID of the page to get.
            if the page is a title, the latest revision will be used.

        Returns:
            BaseModel: The loaded model.
        """
        df = self._get_features(page)

        if self.model_type == ModelType.DeepNeuralNetwork:
            self.model = DeepNeuralNetwork(df, self.features_set)

        elif self.model_type == ModelType.LightGBM:
            self.model = LightGBM(df, self.features_set)
        
        return self.model

    def predict_quality(self, page_title: str | int):
        """
        Predicts the quality of a Wikipedia page.

        Args:
            page (str | int): The title of the page or the revision ID of the page to get.
            if the page is a title, the latest revision will be used.

        Returns:
            str: The predicted quality of the page.
        """
        model = self._load_model(page_title)
        return model.predict()
