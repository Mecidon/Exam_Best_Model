"""
Imports
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, recall_score,
                             precision_score, f1_score)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

class Classifier():
    """
    Class for Classifier models.
    """
    def __init__(self,X, y, test_size, random_state,
                 max_iter, degree, scaling_first):
        self.test_size = test_size
        self.random_state = random_state
        self.max_iter = max_iter
        self.degree = degree
        self.scaling_first = scaling_first

        # Splitting the data
        self.X_train, self.X_test, \
        self.y_train, self.y_test = \
        train_test_split(X,
                         y,
                         test_size=self.test_size,
                         random_state=self.random_state)

    @ignore_warnings(category=[ConvergenceWarning, UserWarning])
    def LoR_Model(self):
        """
        Func to create Logistic Regression Model.
        """
        # Define the steps for the pipeline
        steps = [("scaler", StandardScaler()),
                ("preprocessor", PolynomialFeatures(include_bias=False)),
                ("estimator", LogisticRegression(random_state=self.random_state,
                                                 max_iter=self.max_iter))]

        # Depending on the value of scaling_first, switch the order of steps
        if not self.scaling_first:
            steps[0], steps[1] = steps[1], steps[0]

        # Define the pipeline
        pipe = Pipeline(steps=steps)

        # Define the parameter grid for GridSearchCV
        grid_param = {"preprocessor__degree": np.arange(*self.degree),
                      "estimator__penalty": ["elasticnet"], # Only using elastic for simplicity.
                      "estimator__tol": [0.001, 0.0001],
                      "estimator__C": np.logspace(0, 1, 10),
                      "estimator__class_weight": ["balanced", None],
                      "estimator__solver": ["saga"],
                      "estimator__multi_class": ["auto", "ovr", "multinomial"],
                      "estimator__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
                      }

        # Define GridSearchCV
        grid_model = GridSearchCV(pipe,
                                  grid_param,
                                  cv=10,
                                  scoring="accuracy")

        # Fit the GridSearchCV
        grid_model.fit(self.X_train, self.y_train)

        y_pred = grid_model.predict(self.X_test)
        print(f"\n{grid_model.best_params_}")
        print(confusion_matrix(self.y_test,y_pred))
        accuracy = accuracy_score(self.y_test,y_pred)
        recall = recall_score(self.y_test,y_pred,average="weighted")
        precision = precision_score(self.y_test,y_pred,average="weighted")
        f1 = f1_score(self.y_test,y_pred,average="weighted")
        print(classification_report(self.y_test,y_pred))

        return grid_model, accuracy, recall, precision, f1

    @ignore_warnings(category=[ConvergenceWarning, UserWarning])
    def KNN_Model(self):
        """
        Func to create K Nearest Neighbours Model.
        """
        steps = [("scaler", StandardScaler()),
                ("estimator", KNeighborsClassifier())]

        pipe = Pipeline(steps=steps)

        grid_param = {"estimator__n_neighbors": np.arange(1,21),
                      "estimator__weights": ["uniform", "distance"],
                      "estimator__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                      "estimator__leaf_size": np.arange(1,5),
                    #   "estimator__p": [1,2], # To save time.
                    #   "estimator__metric": ["minkowski", "cityblock", "euclidean", "chebyshev"] # To save time.
                      }

        grid_model = GridSearchCV(pipe,
                                  grid_param,
                                  cv=10,
                                  scoring="accuracy")

        grid_model.fit(self.X_train, self.y_train)

        y_pred = grid_model.predict(self.X_test)
        print(f"\n{grid_model.best_params_}")
        print(confusion_matrix(self.y_test,y_pred))
        accuracy = accuracy_score(self.y_test,y_pred)
        recall = recall_score(self.y_test,y_pred,average="weighted")
        precision = precision_score(self.y_test,y_pred,average="weighted")
        f1 = f1_score(self.y_test,y_pred,average="weighted")
        print(classification_report(self.y_test,y_pred))

        return grid_model, accuracy, recall, precision, f1

    @ignore_warnings(category=[ConvergenceWarning, UserWarning])
    def SVC_Model(self):
        """
        Func to create Support Vector Classifier Model.
        """
        steps = [("scaler", StandardScaler()),
                ("estimator", SVC(max_iter=self.max_iter,
                                  random_state=self.random_state))]

        pipe = Pipeline(steps=steps)

        grid_param = {"estimator__C": np.logspace(0,1,10),
                    #   "estimator__kernel": ["linear", "poly", "rbf", "sigmoid"], # Can't get it working...!
                      "estimator__degree": np.arange(2, 4),
                      "estimator__gamma": ["scale", "auto"],
                      "estimator__class_weight": ["balanced",None]
                      }

        grid_model = GridSearchCV(pipe,
                                  grid_param,
                                  cv=10,
                                  scoring="accuracy")

        grid_model.fit(self.X_train, self.y_train)

        y_pred = grid_model.predict(self.X_test)
        print(f"\n{grid_model.best_params_}")
        print(confusion_matrix(self.y_test,y_pred))
        accuracy = accuracy_score(self.y_test,y_pred)
        recall = recall_score(self.y_test,y_pred,average="weighted")
        precision = precision_score(self.y_test,y_pred,average="weighted")
        f1 = f1_score(self.y_test,y_pred,average="weighted")
        print(classification_report(self.y_test,y_pred))

        return grid_model, accuracy, recall, precision, f1
