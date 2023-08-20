from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from wikipediatools.data.features import FeatureSets
from wikipediatools.models.model import BaseModel
import matplotlib.pyplot as plt
from pathlib import Path
import lightgbm as lgbm
import pandas as pd
import numpy as np
import joblib
import optuna
import shap
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


class LightGBM(BaseModel):
    """
    LightGBM classifier with hyperparameter optimization using Optuna.
    """
    def __init__(
        self,
        data,
        features_set: FeatureSets = FeatureSets.ALL,
        boosting_type="gbdt",
        gpu=False,
        eval_size=0.05,
        target_clm="Target",
        random_state=42,
        n_folds=5,
    ):
        """
        Initialize the LightGBM classifier.

        Args:
            data (pandas.DataFrame): The input dataset.
            features_set (list): List of feature column names.
            boosting_type (str, optional): Type of boosting algorithm. Defaults to "gbdt".
            gpu (bool, optional): Whether to use GPU acceleration. Defaults to True.
            eval_size (float, optional): Proportion of the dataset to use for evaluation. Defaults to 0.05.
            target_clm (str, optional): Name of the target column. Defaults to "Target".
            random_state (int, optional): Random seed. Defaults to 42.
            n_folds (int, optional): Number of folds for cross-validation. Defaults to 5.
        """

        if isinstance(data, pd.DataFrame):
            self.data = data.copy()
        else:
            raise TypeError("data must be of type pandas.DataFrame")
            
        if isinstance(features_set, FeatureSets):
            self.features_set = features_set.value
        elif all(isinstance(x, FeatureSets) for x in features_set):
            results = []
            for feature_set in features_set:
                results.extend(feature_set.value)
            self.features_set = results
        else:    
            raise TypeError("features_set must be of type FeatureSets or a list of FeatureSets")

        self.boosting_type = boosting_type
        self.target_clm = target_clm
        self.random_state = random_state
        self.n_folds = n_folds
        self.device = None
        self._model: lgbm.LGBMClassifier = None
        self._best_study: optuna.Study = None
        self.best_model: lgbm.LGBMClassifier = None
        self.loaded_model :lgbm.LGBMClassifier = None
        self.eval_data = None

        if gpu:
            self.device = "gpu"

        if len(self.data) > 1:
            self.train_data, self.eval_data = train_test_split(
                self.data,
                test_size=eval_size,
                random_state=random_state,
                stratify=self.data[target_clm])
        
        if features_set is FeatureSets.READABILITY_SCORES:
            self.best_model = self.load_model(f"LGBM_models/ReadabilityScoresFeatures/{boosting_type.lower()}/best_model.pkl")
        elif features_set is FeatureSets.STRUCTURE:
            self.best_model = self.load_model(f"LGBM_models/StructureFeatures/{boosting_type.lower()}/best_model.pkl")
        elif features_set is FeatureSets.TEXT_STATISTICS:
            self.best_model = self.load_model(f"LGBM_models/TextStatisticsFeatures/{boosting_type.lower()}/best_model.pkl")
        elif features_set is FeatureSets.WRITING_STYLE:
            self.best_model = self.load_model(f"LGBM_models/WritingStyleFeatures/{boosting_type.lower()}/best_model.pkl")
        elif features_set is FeatureSets.EDIT_HISTORY:
            self.best_model = self.load_model(f"LGBM_models/EditHistoryFeatures/{boosting_type.lower()}/best_model.pkl")
        elif features_set is FeatureSets.NETWORK:
            self.best_model = self.load_model(f"LGBM_models/NetworkFeatures/{boosting_type.lower()}/best_model.pkl")
        elif features_set is FeatureSets.ALL:
            self.best_model = self.load_model(f"LGBM_models/AllFeatures/{boosting_type.lower()}/best_model.pkl")
        elif features_set is FeatureSets._CUSTOM:
            self.best_model = self.load_model(f"LGBM_models/CustomFeatures/{boosting_type.lower()}/best_model.pkl")
    
    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function to be maximized.
        """
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": self.boosting_type,
            "verbose": -1,
            "n_estimators": trial.suggest_int("n_estimators", 100, 4500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 16, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 0.9, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 1024, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1),
            "subsample": trial.suggest_float("subsample", 0.1, 1),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 50),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 300),
            "min_split_gain": trial.suggest_float("min_split_gain", 1e-8, 100, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.1, 1),
        }
        kf = KFold(n_splits=self.n_folds, random_state=self.random_state, shuffle=True)
        model = lgbm.LGBMClassifier(device=self.device, seed= self.random_state, **params)
        accuracies = [None] * self.n_folds

        for i, (train_index, test_index) in enumerate(kf.split(self.train_data)):
            train = self.train_data.iloc[train_index]
            x_train, y_train = train[self.features_set], train[self.target_clm]

            test = self.train_data.iloc[test_index]
            x_test, y_test = test[self.features_set], test[self.target_clm]

            callbacks = [
                lgbm.early_stopping(10, verbose=False),
                lgbm.log_evaluation(period=0),
            ]

            if self.boosting_type == "dart":
                callbacks = callbacks[1:]
            
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_test, y_test)],
                callbacks=callbacks
            )
            
            preds = model.predict(x_test)
            accuracy = accuracy_score(y_test, preds)
            accuracies[i] = accuracy

            self._model = model
            if trial.should_prune():
                raise optuna.TrialPruned()

        results = np.mean(accuracies)
        print(f"Trial {trial.number} - Accuracy: {results}")
        return results

    def _store_best_model(self, study: optuna.Study, trial: optuna.Trial) -> None:
        if self._best_study is not None:
            if study.best_value > self._best_study.best_value:
                self.best_model = self._model
        elif study.best_trial == trial:
            self.best_model = self._model

    def _optimize(self, path, n_studies, n_trials, n_jobs) -> None:
        """
        Perform hyperparameter optimization using Optuna.

        Args:
            path (str): Path to store the trial history.
            n_studies (int): Number of Optuna studies.
            n_trials (int): Number of trials per study.
            n_jobs (int): Number of parallel jobs for Optuna optimization.
        """
        for i in range(n_studies):
            print(f'Hyperparameter optimization for model Study Nr. {i + 1}')
            sampler = optuna.samplers.TPESampler(n_startup_trials=10, seed=self.random_state + i)
            study = optuna.create_study(
                directions=["maximize"],
                sampler=sampler,
                study_name=f"study{i + 1}",
                pruner=optuna.pruners.HyperbandPruner(),
            )

            study.optimize(
                self._objective,
                n_trials=n_trials,
                n_jobs=n_jobs,
                callbacks=[self._store_best_model],
            )
            if path:
                storage = f"LGBM_Storage/{path}/{self.boosting_type}"
                Path(storage).mkdir(parents=True, exist_ok=True)
                result = study.trials_dataframe(attrs=("number", "params", "duration", "value"))
                result["auc"] = result["value"]
                del result["value"]
                result["duration"] = result["duration"] / np.timedelta64(1, "s")
                result = result.sort_values(by="auc", ascending=False)
                result.to_csv(f"{storage}/study{i + 1}.csv", index=False)

            print(f"\nBest trial in study {i + 1} Trial({study.best_trial.number}) - Params: ")
            for key, value in study.best_trial.params.items():
                print(f"    {key}: {value}")

            if self._best_study is None:
                self._best_study= study
            elif study.best_value > self._best_study.best_value:
                self._best_study = study

    def optimize_and_train(self, path=None, n_studies=2, n_trials=100, n_jobs=1, verbose=False) -> lgbm.LGBMClassifier:
        """
        Perform hyperparameter optimization and train the model.

        Args:
            path (str, optional): Path to the training data. Defaults to None. if None, the model will not be saved.
            n_studies (int, optional): Number of Optuna studies. Defaults to 2.
            n_trials (int, optional): Number of trials per study. Defaults to 100.
            n_jobs (int, optional): Number of parallel jobs for Optuna optimization. Defaults to 1.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.

        Returns:
            lgbm.LGBMClassifier: Trained LightGBM classifier.
        """
        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
        self._optimize(path, n_studies, n_trials, n_jobs)
        return self.train(path)

    def train(self, path = None) -> lgbm.LGBMClassifier:
        """
        Train the model using the best hyperparameters found during optimization.

        Args:
            path (str, optional): Path to the training data. Defaults to None.

        Returns:
            lgbm.LGBMClassifier: Trained LightGBM classifier.
        """
        if self.best_model:
            model = self.best_model
        else:
            model = lgbm.LGBMClassifier(device=self.device, verbose=-1, random_state=self.random_state)
        kf = KFold(self.n_folds, random_state=self.random_state, shuffle=True)
        callbacks = [
                lgbm.early_stopping(10, verbose=False),
                lgbm.log_evaluation(period=0),
            ]
        print("\n--Training--")
        for i, (train_index, test_index) in enumerate(kf.split(self.train_data)):
            train = self.train_data.iloc[train_index]
            x_train, y_train = train[self.features_set], train[self.target_clm]

            test = self.train_data.iloc[test_index]
            x_test, y_test = test[self.features_set], test[self.target_clm]
            model.fit(x_train, y_train, eval_metric="auc", eval_set=[(x_test, y_test)], callbacks=callbacks)
            preds = model.predict(x_test)
            print(f"Fold {i} - accuracy: {accuracy_score(preds, y_test)}")
        
        preds = model.predict(self.eval_data[self.features_set])
        actual = self.eval_data[self.target_clm]
        print(f"Evaluation accuracy: {accuracy_score(actual, preds)}")

        if path:
            storage = f"LGBM_Storage/{path}/{self.boosting_type}/"
            Path(storage).mkdir(parents=True, exist_ok=True)
            joblib.dump(model, f"{storage}/best_model.pkl")

        return model

    def get_params_from_df(self, path) -> dict:
        """
        Retrieve the best hyperparameters from a results DataFrame.

        Args:
            path (str): Path to the results DataFrame CSV file.

        Returns:
            dict: Best hyperparameters.
        """
        temp = pd.read_csv(path)
        temp_params = temp.iloc[0][1:-2].to_dict()
        params = {}
        for k, v in temp_params.items():
            k = k.split("params_")[1]
            if k == "n_estimators":
                v = int(v)
            elif k == "num_leaves":
                v = int(v)
            elif k == "min_child_samples":
                v = int(v)
            elif k == "subsample_freq":
                v = int(v)
            elif k == "max_depth":
                v = int(v)
            elif k == "bagging_freq":
                v = int(v)
            params[k] = v
        return params

    def train_custom_model(self, params: dict) -> lgbm.LGBMClassifier:
        """
        Train a custom LightGBM model with specified hyperparameters.

        Args:
            params (dict): Hyperparameters for the LightGBM model.

        Returns:
            lgbm.LGBMClassifier: Trained LightGBM classifier.
        """
        model = lgbm.LGBMClassifier(device=self.device, verbose=-1, **params)
        kf = KFold(self.n_folds, random_state=self.random_state, shuffle=True)
        for i, (train_index, test_index) in enumerate(kf.split(self.train_data)):
            train = self.train_data.iloc[train_index]
            x_train, y_train = train[self.features_set], train[self.target_clm]

            test = self.train_data.iloc[test_index]
            x_test, y_test = test[self.features_set], test[self.target_clm]
            model.fit(x_train, y_train)
            preds = model.predict(x_test)
            print(f"Fold {i} - accuracy: {accuracy_score(preds, y_test)}")

        preds = model.predict(self.eval_data[self.features_set])
        actual = self.eval_data[self.target_clm]
        print(f"Evaluation accuracy: {accuracy_score(actual, preds)}")
        return model
    
    def load_model(self, path) -> lgbm.LGBMClassifier :
        """
        Load a trained LightGBM model from a file.

        Args:
            path (str): Path to the model file.

        Returns:
            lgbm.LGBMClassifier: Loaded LightGBM model.
        """
        self.loaded_model = joblib.load(path) 
        return self.loaded_model
    
    def evaluate(self, model: lgbm.LGBMClassifier = None) -> float:
        """
        Test a LightGBM model on the evaluation dataset.

        Args:
            model (lgbm.LGBMClassifier): Pre-trained LightGBM model. If None, the previously loaded model will be used.

        Returns:
            float: Evaluation accuracy score.
        """

        if model is None:
            model = self._get_model()

        # Make predictions on the evaluation data
        preds = model.predict(self.eval_data[self.features_set])
        actual = self.eval_data[self.target_clm]

        # Calculate the accuracy score
        accuracy = accuracy_score(actual, preds)

        # Print the evaluation accuracy
        print(f"Evaluation accuracy: {accuracy}")

        return accuracy

    def _get_model(self):
        if self.loaded_model is None:
            model = self.best_model
        else:
            model = self.loaded_model
        return model

    def predict(self):
        """
        Predict the quality of a Wikipedia article.
        """
        return self.best_model.predict(self.data[self.features_set].values, verbose=-1)[0]
    
    def get_params(self):
        """
        Get the best hyperparameters of the model.
        """
        return self.best_model.get_params()
    
    def feature_importance(self, plot_type="bar", index: int=None):
        """
        Plot the feature importance of the model.
        Args:
            plot_type (str, optional): The type of plot to use. Defaults to "bar".
            index (int, optional): The index of the sample to explain. Defaults to None.
        Returns:
            matplotlib.figure.Figure: The figure of the plot.
            matplotlib.axes.Axes: The axes of the plot.
        """ 
        if self.eval_data is None:
            raise ValueError("The input data must be larger than 10")
        
        data = self.eval_data[self.features_set].values
        explainer = shap.Explainer(self.best_model, data)
        explainer.feature_names = self.features_set
        shap_values = explainer(data, check_additivity=False)
        if index and plot_type == "bar":
            shap.plots.bar(shap_values[index], max_display=12, show=False)
        else:
            shap.summary_plot(shap_values,
                              data,
                              max_display=99,
                              plot_type=plot_type,
                              feature_names=self.features_set,
                              show=False, plot_size=None, color="red")
        return plt.gcf(), plt.gca()
    
    def classification_metrics(self, model: lgbm.LGBMClassifier = None, data: pd.DataFrame = None):
        if model is None:
            model = self._get_model()
        if data is None:
            data = self.eval_data
        # Make predictions on the evaluation data
        preds = pd.Categorical(model.predict(data[self.features_set])).codes.astype("float32")
        actual = pd.Categorical(data[self.target_clm]).codes.astype("float32")
        # Calculate the accuracy score
        accuracy = accuracy_score(actual, preds)
        precision = precision_score(actual, preds)
        recall = recall_score(actual, preds)
        f1 = 2 * (precision * recall) / (precision + recall)
        tn, fp, fn, tp = confusion_matrix(actual, preds).ravel()
        tnr = tn / (tn + fp)



        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"True Negative Rate: {tnr}")
        print(f"F1: {f1}")
        
        return accuracy, precision, recall, f1, tnr