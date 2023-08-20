from keras.metrics import Precision, Recall, TrueNegatives, FalsePositives
from sklearn.model_selection import train_test_split
from keras_tuner import BayesianOptimization
from sklearn.model_selection import KFold
from wikipediatools.data.features import FeatureSets
from keras_tuner import HyperParameters
from keras.layers import Dense, Dropout
from matplotlib import pyplot as plt
from keras.models import load_model
from keras.models import Sequential
from models.model import BaseModel
import tensorflow as tf
import pandas as pd
import shap
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class DeepNeuralNetwork(BaseModel):
    """
    A class used to represent a Deep Neural Network model.
    """

    def __init__(self, data: pd.DataFrame,
                features_set: FeatureSets = FeatureSets.ALL,
                 gpu=False,
                 eval_size=0.05,
                 target_clm="Target",
                 random_state=42) -> None:
        """
        Initializes the model.
        Args:
            data (pandas.DataFrame): The dataset to train the model on.
            features_set (FeatureSets, optional): The features to use. Defaults to FeatureSets.ALL.
            gpu (bool, optional): Whether to use GPU or not. Defaults to False.
            eval_size (float, optional): The size of the evaluation set. Defaults to 0.05.
            target_clm (str, optional): The name of the target column. Defaults to "Target".
            random_state (int, optional): The random state to use. Defaults to 42.
        """
        if isinstance(data, pd.DataFrame):
            self.data = data.copy()
        else:
            raise TypeError("data must be a pandas.DataFrame")
        
        if isinstance(features_set, FeatureSets):
            self.features_set = features_set.value
        elif all(isinstance(x, FeatureSets) for x in features_set):
            results = []
            for feature_set in features_set:
                results.extend(feature_set.value)
            self.features_set = results
        else:
            raise TypeError("features_set must be an instance of FeaturesSet Enum or a list of FeaturesSet Enum")
        
        self.eval_size = eval_size
        self.target_clm = target_clm
        self.random_state = random_state
        self.best_model: Sequential = None

        if features_set is FeatureSets.TEXT_STATISTICS:
            self.best_model = load_model("DNN_models/TextStatsiticsFeatures.h5")
        elif features_set is FeatureSets.STRUCTURE:
            self.best_model = load_model("DNN_models/StructureFeatures.h5")
        elif features_set is FeatureSets.READABILITY_SCORES:
            self.best_model = load_model("DNN_models/ReadabilityScoresFeatures.h5")
        elif features_set is FeatureSets.WRITING_STYLE:
            self.best_model = load_model("DNN_models/WritingStyleFeatures.h5")
        elif features_set is FeatureSets.EDIT_HISTORY:
            self.best_model = load_model("DNN_models/EditHistoryFeatures.h5")
        elif features_set is FeatureSets.NETWORK:
            self.best_model = load_model("DNN_models/NetworkFeatures.h5")
        elif features_set is FeatureSets.ALL:
            self.best_model = load_model("DNN_models/AllFeatures.h5")

        data[target_clm] = pd.Categorical(data[target_clm]).codes
        if not gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        self.eval_data = None
        if len(data) > 1:
            self.train_data, self.eval_data = train_test_split(
                data,
                test_size=eval_size,
                random_state=random_state,
                stratify=data[target_clm])
            

    def _base_model(self):
        """
        Creates the base model.
        Returns:
            Sequential: The base model.
        """
        model = Sequential()
        model.add(Dense(100, input_dim=len(self.features_set),
                        activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=["accuracy"])
        return model

    def _create_model(self, hp: HyperParameters) -> Sequential:
        """
        Creates a model with hyperparameters to be tuned.
        Args:
            hp (HyperParameters): The hyperparameters to use.
        Returns:
            Sequential: The model to be tuned.
        """
        model = Sequential()
        n_layers = hp.Int('num_layers', 2, 5)
        l1_l2 = hp.Choice('l1_l2', values=[0.1, 0.01, 0.001, 0.0001])
        l1_l2 = tf.keras.regularizers.l1_l2(l1=l1_l2, l2=l1_l2)
        for i in range(n_layers):
            if i < n_layers - 1:
                units = hp.Int('units_' + str(i), 10, 100, 5)
                model.add(Dense(units,
                                activation='relu',
                                kernel_regularizer=l1_l2))
                model.add(Dropout(0.2))
            else:
                model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=["accuracy"])
        return model

    def optimize_and_train(self, path, n_trials=200, epochs=195, patience=25, verbose=1, n_folds=5) -> Sequential:
        """Optimize and train the model using Keras Tuner
        Args:
            path (str): The path to save the model to.
            n_trials (int, optional): The number of trials to run. Defaults to 200.
            epochs (int, optional): The number of epochs to train the model for. Defaults to 195.
            patience (int, optional): The number of epochs to wait before early stopping. Defaults to 25.
            verbose (int, optional): The verbosity level. Defaults to 1.
            n_folds (int, optional): The number of folds to use for cross validation. Defaults to 5.
            Returns:
                Sequential: The best model
        """
        x_train = self.train_data[self.features_set].values
        y_train = self.train_data[self.target_clm].values
        start = time.time()
        tuner = BayesianOptimization(self._create_model,
                                     objective='accuracy',
                                     max_trials=n_trials,
                                     overwrite=True,
                                     seed=self.random_state)
        callbacks = [tf.keras.callbacks.EarlyStopping(
            patience=patience, monitor='accuracy')]
        tuner.search(x_train, y_train, epochs=epochs,
                     callbacks=callbacks, verbose=verbose)
        best_model: Sequential = tuner.get_best_models()[0]

        best_model = self.train(model=best_model,
                   epochs=epochs,
                   patience=patience,
                   verbose=verbose,
                   n_folds=n_folds,
                   cv=True)

        best_model.save(f"{path}.h5")
        print("--- %s seconds ---" % (time.time() - start))
        self.best_model = best_model
        return best_model

    def train(self, model = None, epochs=195, patience=25, verbose=1, n_folds=5, cv=True) -> Sequential:
        """
        Train the model using Keras Tuner

        Args:
            model (Sequential, optional): The model to train. Defaults to the best model.
            epochs (int, optional): The number of epochs to train the model for. Defaults to 195.
            patience (int, optional): The number of epochs to wait before early stopping. Defaults to 25.
            verbose (int, optional): The verbosity level. Defaults to 1.
            n_folds (int, optional): The number of folds to use for cross validation. Defaults to 5.
            cv (bool, optional): Whether to use cross validation or not. Defaults to False.
        Returns:
                Sequential: The best model
        """
        if model is None and self.best_model is None:
            model = self._base_model()
        if model is None:
            model = self.best_model

        callbacks = [tf.keras.callbacks.EarlyStopping(
            patience=patience, monitor='accuracy')]

        if cv:
            kf = KFold(n_splits=n_folds,
                       random_state=self.random_state, shuffle=True)
            
            for train_index, test_index in kf.split(self.train_data):
                train = self.train_data.iloc[train_index]
                x, y = train[self.features_set], train[self.target_clm]

                test = self.train_data.iloc[test_index]
                x_test, y_test = test[self.features_set], test[self.target_clm]

                model.fit(x, y,
                          epochs=epochs,
                          validation_data=(x_test, y_test),
                          callbacks=callbacks,
                          verbose=verbose,
                          batch_size=15)
        else:
            x_train = self.train_data[self.features_set].values
            y_train = self.train_data[self.target_clm].values
            
            model.fit(x_train, y_train,
                      epochs=epochs,
                      callbacks=callbacks,
                      verbose=verbose)

        return model
    

    def evaluate(self, model: Sequential = None, data: pd.DataFrame = None) -> float:
        """
        Evaluate the model on the evaluation data.

        Args:
            model (Sequential, optional): The model to evaluate. Defaults to None.
        Returns:
                float: The accuracy of the model.
                
        """
        if model is None:
            model = self.best_model

        if data is None:
            data = self.eval_data
        x_eval = data[self.features_set].values
        y_eval = data[self.target_clm].values
        accuracy = model.evaluate(x_eval, y_eval, verbose=0)[1]
        print(f"Evaluation accuracy: {accuracy}")
        return model.evaluate(x_eval, y_eval, verbose=0)
    
    def predict(self):
        """
            Predict the quality of a Wikipedia article.
        """
        data = self.data[self.features_set].values.astype('float32')
        pred = self.best_model(data)
        return "High" if pred <=0.5 else "Low"
    
    def get_params(self):
        """
            Get the best hyperparameters of the model.
        """
        return self.best_model.get_config()
    
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
        shap_values = explainer(data)
        if index and plot_type == "bar":
            shap.plots.bar(shap_values[index], max_display=12, show=False)
        else:
            shap.summary_plot(shap_values,
                              data,
                              max_display=12,
                              plot_type=plot_type,
                              feature_names=self.features_set,
                              show=False, plot_size=None)
        return plt.gcf(), plt.gca()

    def summary(self):
        """
        Print the summary of the model.
        """
        return self.best_model.summary()
    
    def classification_metrics(self, model: Sequential = None, data: pd.DataFrame = None):

        if model is None:
            model = self.best_model

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy" , Precision(), Recall(), TrueNegatives(), FalsePositives()])
        
        loss, accuracy, precision, recall, tn, fp = self.evaluate(model, data)
        f1 = 2 * (precision * recall) / (precision + recall)
        tnr = tn / (tn + fp)
        print(f"Loss: {loss}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"True Negative Rate: {tnr}")
        print(f"F1 Score: {f1}")
        return accuracy, precision, recall, f1, tnr