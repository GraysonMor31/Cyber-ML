'''
Custom PyTorch module for Random Forest Classification
'''

# Import Dependencies
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class RandomForestPreclassifier:
    def __init__(self, n_estimators=100, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X_train, y_train):
        """
        Train the Random Forest classifier on the training data.

        Parameters:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
        """
        self.rf.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Make predictions on the test data using the trained Random Forest model.

        Parameters:
            X_test (pd.DataFrame): Test features.

        Returns:
            pd.Series: Predicted labels for the test data.
        """
        return self.rf.predict(X_test)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the performance of the Random Forest classifier on the test data.

        Parameters:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test labels.

        Returns:
            dict: Evaluation metrics (accuracy, precision, recall, F1-score).
        """
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)

        return {
            'accuracy': accuracy,
            'classification_report': report
        }