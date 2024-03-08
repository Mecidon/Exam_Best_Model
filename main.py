"""
Imports
"""
import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.utils.multiclass import type_of_target
from validation import Validation as Val
from regressor import Regressor as Reg
from classifier import Classifier as Cla

############################################################################################################################
# 0. USER GUIDE                                                                                                            #
# 1. Enter csv such as Advertising.csv                                                                                     #
# 2. Enter number for target column                                                                                        #
# 3. If string columns in X, enter column number to convert, then ["continue","Continue","CONTINUE"]                       #
# 4. Enter number for regressor/ classifier                                                                                #
# 5. test_size      as float such as 0.3                                                                                   #
# 6. random_state   as integer such as 101                                                                                 #
# 7. max_inter      as integer such as 10000                                                                               #
# 8. degree         as two ints such as 1,2 (faster)                                                                       #
# 9. Choose scaling BEFORE poly as ["TRUE", "True", "true", "t"] or scaling AFTER poly as ["FALSE", "False", "false", "f"] #
############################################################################################################################

def read_csv_file():
    """
    Func to read in a name of existing csv file in directory.
    """
    while True:
        filename_string = input("\nEnter CSV filename: ")
        if os.path.isfile(filename_string):
            print("\nFile Found.")
            return filename_string
        else:
            print("File not found. Please try again.")

def choose_label(filename_string):
    """
    Func to read in a CSV file, print all columns as a numbered list,
    choose Label by number and create df, X, y.
    """
    df = pd.read_csv(filename_string)
    # Not Used For Now (To downsize filesize wherever possible)
    # df = df.apply(pd.to_numeric, errors='coerce', downcast='integer')
    # df = df.apply(pd.to_numeric, errors='coerce', downcast='float')
    print("\nColumns in the DataFrame:")
    for i, column in enumerate(df.columns, start=1):
        print(f"{i}. {column}")

    while True:
        try:
            choice = int(input("\nEnter the column number used as y Label: "))
            if 1 <= choice <= len(df.columns):
                y_column = df.columns[choice - 1]
                y = df[y_column]
                if y.dtype == 'object' and any(y.str.isnumeric()):
                    print("\nError: Selected column contains both string and numeric values.")
                    continue
                elif y.isnull().values.any():
                    print("Error: Selected column contains NaN values.")
                    continue
                X = df.drop(y_column, axis=1)
                print(f"Selected column '{y_column}' as the y label.")
                return X, y
            else:
                print("Invalid choice. Enter a number within the range.")
        except ValueError:
            print("Invalid input. Enter an existing number.")

def choose_regressor_classifier(y):
    """
    Func to choose Regressor or Classifier, then validate the choice.
    """
    # Validate type of target
    target_type = type_of_target(y)
    if target_type == "continuous":
        valid_choices = {"1": "Regressor"}
    elif target_type == "binary":
        valid_choices = {"2": "Classifier"}
    elif target_type == "multiclass":
        valid_choices = {"2": "Classifier"}
    else:
        print("The selected y label is of unknown type.")

    while True:
        print("\nChoose:\n1. Regressor\n2. Classifier")
        choice = input("\nEnter choice as number 1 or 2: ")
        if choice in valid_choices:
            chosen_option = valid_choices[choice]
            print(f"You chose {chosen_option}.")
            break
        else:
            print("Invalid choice type of the y label. Please Try again.")

    return target_type

def data_check(X):
    """
    Func to check if:
    A, data is missing values - if so exit.
    B, data has object columns - if so offer to convert to dummies.
    C, if none of the above or finished with B - continue.
    """
    # A
    if X.isnull().values.any():
        missing_rows = X[X.isnull().any(axis=1)].index.tolist()
        # + 2 to have correct .csv row.
        adjusted_rows = [row + 2 for row in missing_rows]
        print("\nMissing data found in rows: "+\
              f"{', '.join(map(str, adjusted_rows))}. "+\
                "Please correct it before proceeding.")
        print("Exiting Program...")
        sys.exit()
    # B
    elif not X.select_dtypes(include=['object']).empty:
        print("\nNo missing data found.")
        object_columns = X.select_dtypes(include=['object']).columns.tolist()
        print("\nObject type columns found:")
        for index, column in enumerate(object_columns, start=1):
            print(f"{index}. {column}")

        while True:
            choice = input("\nEnter the column number to convert to "+\
                        "dummy variables, or 'continue' to proceed: ")
            if choice in ["continue","Continue","CONTINUE"]:
                return X
            try:
                choice = int(choice)
                if 1 <= choice <= len(object_columns):
                    selected_column = object_columns[choice - 1]
                    print(f"Selected column '{selected_column}' "+\
                        "to convert to dummy variables.")
                    X = pd.get_dummies(X, columns=[selected_column],
                                    drop_first=True, dtype="int8")
                    object_columns.remove(selected_column)
                    print("\nRemaining object-type columns:")
                    for i, column in enumerate(object_columns, start=1):
                        print(f"{i}. {column}")
                    print(X)
                else:
                    print("Invalid choice. Enter a number within "+\
                        "the range or 'continue'.")
            except ValueError:
                print("Invalid input. Enter a valid number or 'continue'.")
    # C
    else:
        print("\nNo missing data found.")
        print("No object type columns found.\n")
        return X

def modeling(target_type, X, y, test_size, random_state,
             max_iter, degree, scaling_first):
    """
    Func to run the models from Regressor/ Classifier classes with timer.
    """
    results = {}

    if target_type in ["continuous"]:
        regressor_instance = Reg(X, y, test_size, random_state, max_iter, degree, scaling_first)

        # LiR Model
        start_time = time.time()
        lir_model, lir_MAE, lir_RMSE, lir_r2 = regressor_instance.LiR_Model()
        end_time = time.time()
        total_runtime = end_time - start_time
        print(f"LiR Runtime: {total_runtime:.2f} seconds\n")
        results["LinearRegression"] = (lir_model, lir_MAE, lir_RMSE, lir_r2)

        # Lasso Model
        start_time = time.time()
        lasso_model, lasso_MAE, lasso_RMSE, lasso_r2 = regressor_instance.Lasso_Model()
        end_time = time.time()
        total_runtime = end_time - start_time
        print(f"Lasso Runtime: {total_runtime:.2f} seconds\n")
        results["Lasso"] = (lasso_model, lasso_MAE, lasso_RMSE, lasso_r2)

        # Ridge Model
        start_time = time.time()
        ridge_model, ridge_MAE, ridge_RMSE, ridge_r2 = regressor_instance.Ridge_Model()
        end_time = time.time()
        total_runtime = end_time - start_time
        print(f"Ridge Runtime: {total_runtime:.2f} seconds\n")
        results["Ridge"] = (ridge_model, ridge_MAE, ridge_RMSE, ridge_r2)

        # Elastic Model
        start_time = time.time()
        elastic_model, elastic_MAE, elastic_RMSE, elastic_r2 = regressor_instance.Elastic_Model()
        end_time = time.time()
        total_runtime = end_time - start_time
        print(f"ElasticNet Runtime: {total_runtime:.2f} seconds\n")
        results["ElasticNet"] = (elastic_model, elastic_MAE, elastic_RMSE, elastic_r2)

        # SVR Model
        start_time = time.time()
        svr_model, svr_MAE, svr_RMSE, svr_r2 = regressor_instance.SVR_Model()
        end_time = time.time()
        total_runtime = end_time - start_time
        print(f"SVR Runtime: {total_runtime:.2f} seconds\n")
        results["SVR"] = (svr_model, svr_MAE, svr_RMSE, svr_r2)

        return results

    elif target_type in ["binary", "multiclass"]:
        classifier_instance = Cla(X, y, test_size, random_state, max_iter, degree, scaling_first)

        # LoR Model
        start_time = time.time()
        lor_model, lor_acc, lor_rec, lor_pre, lor_f1 = classifier_instance.LoR_Model()
        end_time = time.time()
        total_runtime = end_time - start_time
        print(f"LoR Runtime: {total_runtime:.2f} seconds\n")
        results["LogisticRegression"] = (lor_model, lor_acc, lor_rec, lor_pre, lor_f1)

        # KNN Model
        start_time = time.time()
        knn_model, knn_acc, knn_rec, knn_pre, knn_f1 = classifier_instance.KNN_Model()
        end_time = time.time()
        total_runtime = end_time - start_time
        print(f"KNN Runtime: {total_runtime:.2f} seconds\n")
        results["KNN"] = (knn_model, knn_acc, knn_rec, knn_pre, knn_f1)

        # SVC Model
        start_time = time.time()
        svc_model, svc_acc, svc_rec, svc_pre, svc_f1 = classifier_instance.SVC_Model()
        end_time = time.time()
        total_runtime = end_time - start_time
        print(f"SVC Runtime: {total_runtime:.2f} seconds\n")
        results["SVC"] = (svc_model, svc_acc, svc_rec, svc_pre, svc_f1)

        return results

    else:
        print("Something has gone wrong. Unknown target type.")

    return results

def best_model_func(models_results, target_type):
    """
    Func to calculate best model for Regressor or Classifier.
    """

    best_model_name = None

    # Initiating score for Regressors
    if target_type == "continuous":
        best_metrics = {
            "MAE": float('inf'),
            "RMSE": float('inf'),
            "R2": -float('inf')
        }
    # Initiating score for Classifiers
    elif target_type in ["binary", "multiclass"]:
        best_metrics = {
            "Accuracy": 0,
            "Recall": 0,
            "Precision": 0,
            "F1": 0
        }
    else:
        print("Invalid target type.")
        return

    for model_name, results in models_results.items():

        # Regressor scores
        if target_type == "continuous":
            _, MAE, RMSE, r2 = results
            # Compare model MAE scores
            if MAE < best_metrics["MAE"]:
                best_metrics["MAE"] = MAE
                best_model_name = model_name
            # Compare model RMSE scores
            if RMSE < best_metrics["RMSE"]:
                best_metrics["RMSE"] = RMSE
                best_model_name = model_name
            # Compare model r2 scores
            if r2 > best_metrics["R2"]:
                best_metrics["R2"] = r2
                best_model_name = model_name

        # Classifier scores
        elif target_type in ["binary", "multiclass"]:
            _, accuracy, recall, precision, f1 = results

            # Calculate the mean to display for best model in terminal
            mean_accuracy = np.mean(accuracy)
            mean_recall = np.mean(recall)
            mean_precision = np.mean(precision)
            mean_f1 = np.mean(f1)

            # Compare models based on mean metrics
            if mean_accuracy > best_metrics["Accuracy"]:
                best_metrics["Accuracy"] = mean_accuracy
                best_model_name = model_name
            if mean_recall > best_metrics["Recall"]:
                best_metrics["Recall"] = mean_recall
                best_model_name = model_name
            if mean_precision > best_metrics["Precision"]:
                best_metrics["Precision"] = mean_precision
                best_model_name = model_name
            if mean_f1 > best_metrics["F1"]:
                best_metrics["F1"] = mean_f1
                best_model_name = model_name

    print(f"Best model: {best_model_name}")
    if target_type == "continuous":
        print(f"MAE: {best_metrics['MAE']}")
        print(f"RMSE: {best_metrics['RMSE']}")
        print(f"R2: {best_metrics['R2']}\n")

    elif target_type in ["binary", "multiclass"]:
        print(f"Accuracy: {best_metrics['Accuracy']}")
        print(f"Recall: {best_metrics['Recall']}")
        print(f"Precision: {best_metrics['Precision']}")
        print(f"F1: {best_metrics['F1']}\n")

    choice = input("Choose:\n1. Dump best model\n2. Dump another model\n")

    if choice == "1":
        # Save the best model
        best_model, *_ = models_results[best_model_name]
        save_best_model(best_model, best_model_name)
    elif choice == "2":
        # List available models
        print("Available models:")
        for index, model_name in enumerate(models_results.keys(), start=1):
            print(f"{index}. {model_name}")
        model_choice = input("Choose the model number to dump: ")
        model_choice = int(model_choice)
        if 1 <= model_choice <= len(models_results):
            model_names = list(models_results.keys())
            model_to_dump_name = model_names[model_choice - 1]
            model_to_dump, *_ = models_results[model_to_dump_name]
            save_best_model(model_to_dump, model_to_dump_name)
        else:
            print("Invalid choice. Please try again.")
    else:
        print("Invalid choice. Please try again.")

def save_best_model(best_model, model_name):
    """
    Func to save the best model as a dump with joblib.
    """
    # Get current date in yy_mm_dd format
    current_date = datetime.datetime.now().strftime("%y_%m_%d")
    final_model_name = input(f"\nName of your {model_name} file (Date automatic): ")
    model_name_with_date = f"{final_model_name}_{current_date}"
    dump(best_model, f"{model_name_with_date}.joblib")
    print(f"\nBest model; {model_name_with_date} saved successfully.\n")

def run():
    """
    Func to run the program with all other functions.
    """
    filename_string = read_csv_file()
    original_X, y = choose_label(filename_string)
    target_type = choose_regressor_classifier(y)
    X = data_check(original_X)

    # Inputs for my classes
    test_size = Val.read_in_float(Val.validate_test_size, "TEST_SIZE: ")
    random_state = Val.read_in_integer(Val.validate_int, "RANDOM_STATE: ")
    max_iter = Val.read_in_integer(Val.validate_int, "MAX_ITER: ")
    degree = Val.read_in_integer_tuple(Val.validate_polynomial_degree, "DEGREE as x,y: ")
    scaling_first = Val.read_in_bool("TRUE for scaling before Poly or FALSE for Poly before Scaling: ")

    start_time = time.time()
    models_results = modeling(target_type, X, y, test_size,random_state,
                              max_iter, degree, scaling_first)
    end_time = time.time()
    total_runtime = end_time - start_time
    print(f"Total runtime for all models: {total_runtime:.2f} seconds\n")

    best_model_func(models_results, target_type)

if __name__ == "__main__":
    run()
