"""
Imports
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

class Regressor():
    """
    Class for Regressor models.
    """
    def __init__(self, X, y, test_size, random_state,
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

    @ignore_warnings(category=ConvergenceWarning)
    def LiR_Model(self):
        """
        Func to create Linear Regression Model.
        """
        # Define the steps for the pipeline
        steps = [("scaler", StandardScaler()),
                 ("preprocessor", PolynomialFeatures(include_bias=False)),
                 ("estimator", LinearRegression())]

        # Depending on the value of scaling_first, switch the order of steps
        if not self.scaling_first:
            steps[0], steps[1] = steps[1], steps[0]

        # Define the pipeline
        pipe = Pipeline(steps=steps)

        # Define the parameter grid for GridSearchCV
        grid_param = {"preprocessor__degree": np.arange(*self.degree)}

        # Define GridSearchCV
        grid_model = GridSearchCV(pipe,
                                  grid_param,
                                  cv=10,
                                  scoring="neg_mean_squared_error")

        # Fit the GridSearchCV
        grid_model.fit(self.X_train, self.y_train)

        y_pred = grid_model.predict(self.X_test)
        print(f"\n{grid_model.best_params_}")
        MAE = mean_absolute_error(self.y_test,y_pred)
        print(f"LiR MAE: {MAE}")
        RMSE = np.sqrt(mean_squared_error(self.y_test,y_pred))
        print(f"LiR RMSE: {RMSE}")
        r2 = r2_score(self.y_test,y_pred)
        print(f"LiR r2: {r2}")
        return grid_model, MAE, RMSE, r2

    @ignore_warnings(category=ConvergenceWarning)
    def Lasso_Model(self):
        """
        Func to create l1 Lasso Model.
        """
        steps = [("scaler", StandardScaler()),
                 ("preprocessor", PolynomialFeatures(include_bias=False)),
                 ("estimator", Lasso(max_iter=self.max_iter,
                                     random_state=self.random_state))]

        if not self.scaling_first:
            steps[0], steps[1] = steps[1], steps[0]

        pipe = Pipeline(steps=steps)

        grid_param = {"preprocessor__degree": np.arange(*self.degree),
                      "estimator__alpha": [0.01,0.1,1,10,20,30,50,70,100],
                      "estimator__tol": [0.001, 0.0001],
                      "estimator__selection": ["cyclic", "random"]}

        grid_model = GridSearchCV(pipe,
                                  grid_param,
                                  cv=10,
                                  scoring="neg_mean_squared_error")

        grid_model.fit(self.X_train, self.y_train)

        y_pred = grid_model.predict(self.X_test)
        print(grid_model.best_params_)
        MAE = mean_absolute_error(self.y_test,y_pred)
        print(f"Lasso MAE: {MAE}")
        RMSE = np.sqrt(mean_squared_error(self.y_test,y_pred))
        print(f"Lasso RMSE: {RMSE}")
        r2 = r2_score(self.y_test,y_pred)
        print(f"Lasso r2: {r2}")
        return grid_model, MAE, RMSE, r2

    @ignore_warnings(category=ConvergenceWarning)
    def Ridge_Model(self):
        """
        Func to create L2 Ridge Model.
        """
        steps = [("scaler", StandardScaler()),
                 ("preprocessor", PolynomialFeatures(include_bias=False)),
                 ("estimator", Ridge(max_iter=self.max_iter,
                                     random_state=self.random_state))]

        if not self.scaling_first:
            steps[0], steps[1] = steps[1], steps[0]

        pipe = Pipeline(steps=steps)

        grid_param = {"preprocessor__degree": np.arange(*self.degree),
                      "estimator__alpha": [0.01,0.1,1,10,20,30,50,70,100],
                      "estimator__tol": [0.001, 0.0001],
                      "estimator__solver": ["auto", "svd", "cholesky", "lsqr",\
                                            "sparse_cg", "sag", "saga"]}

        grid_model = GridSearchCV(pipe,
                                  grid_param,
                                  cv=10,
                                  scoring="neg_mean_squared_error")

        grid_model.fit(self.X_train, self.y_train)

        y_pred = grid_model.predict(self.X_test)
        print(grid_model.best_params_)
        MAE = mean_absolute_error(self.y_test,y_pred)
        print(f"Ridge MAE: {MAE}")
        RMSE = np.sqrt(mean_squared_error(self.y_test,y_pred))
        print(f"Ridge RMSE: {RMSE}")
        r2 = r2_score(self.y_test,y_pred)
        print(f"Ridge r2: {r2}")
        return grid_model, MAE, RMSE, r2

    @ignore_warnings(category=ConvergenceWarning)
    def Elastic_Model(self):
        """
        Func to create ElasticNet Model.
        """
        steps = [("scaler", StandardScaler()),
                 ("preprocessor", PolynomialFeatures(include_bias=False)),
                 ("estimator", ElasticNet(max_iter=self.max_iter,
                                          random_state=self.random_state))]

        if not self.scaling_first:
            steps[0], steps[1] = steps[1], steps[0]

        pipe = Pipeline(steps=steps)

        grid_param = {"preprocessor__degree": np.arange(*self.degree),
                      "estimator__alpha": [0.01,0.1,1,10,20,30,50,70,100],
                      "estimator__tol": [0.001, 0.0001],
                      "estimator__l1_ratio": [0.1, 0.5, 0.7, 0.9, 1.0],
                      "estimator__selection": ["cyclic", "random"]}

        grid_model = GridSearchCV(pipe,
                                  grid_param,
                                  cv=10,
                                  scoring="neg_mean_squared_error")

        grid_model.fit(self.X_train, self.y_train)

        y_pred = grid_model.predict(self.X_test)
        print(grid_model.best_params_)
        MAE = mean_absolute_error(self.y_test,y_pred)
        print(f"Elastic MAE: {MAE}")
        RMSE = np.sqrt(mean_squared_error(self.y_test,y_pred))
        print(f"Elastic RMSE: {RMSE}")
        r2 = r2_score(self.y_test,y_pred)
        print(f"Elastic r2: {r2}")
        return grid_model, MAE, RMSE, r2

    @ignore_warnings(category=ConvergenceWarning)
    def SVR_Model(self):
        """
        Func to create Support Vector Regressor Model.
        """
        steps = [("scaler", StandardScaler()),
                 ("estimator", SVR(max_iter=self.max_iter))]

        pipe = Pipeline(steps=steps)

        grid_param = {"estimator__kernel": ["linear", "poly", "rbf", "sigmoid"],
                      "estimator__degree": [*self.degree],
                      "estimator__gamma": ["scale", "auto"],
                    #   "estimator__coef0": [0.0, 0.5, 1.0], # To save time
                      "estimator__tol": [0.001, 0.0001],
                      "estimator__C": np.logspace(0,1,10),
                      "estimator__epsilon": [0.001,0.01,0.1,1,5],
                    #   "estimator__shrinking": [True, False]
                      }

        grid_model = GridSearchCV(pipe,
                                  grid_param,
                                  cv=10,
                                  scoring="neg_mean_squared_error")

        grid_model.fit(self.X_train, self.y_train)

        y_pred = grid_model.predict(self.X_test)
        print(grid_model.best_params_)
        MAE = mean_absolute_error(self.y_test,y_pred)
        print(f"SVR MAE: {MAE}")
        RMSE = np.sqrt(mean_squared_error(self.y_test,y_pred))
        print(f"SVR RMSE: {RMSE}")
        r2 = r2_score(self.y_test,y_pred)
        print(f"SVR r2: {r2}")
        return grid_model, MAE, RMSE, r2
