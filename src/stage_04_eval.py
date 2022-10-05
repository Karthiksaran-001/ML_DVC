from ast import arg
from src.utils.all_utils import read_yaml , create_directory , save_reports
import argparse
import pandas as pd
import os
from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score
import joblib
import numpy as np
import json

def evaluate_metrics(actual_values, predicted_values):
    rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
    mae = mean_absolute_error(actual_values, predicted_values)
    r2 = r2_score(actual_values, predicted_values)
    return rmse, mae, r2

def eval(config_path):
    config = read_yaml(config_path)
    # params = read_yaml(params_path)
    
    ## Read the Train Data
    artifacts_dir = config["artifacts"]['artifacts_dir']
    split_data_folder = config["artifacts"]["split_data_dir"]  
    test_data_file = config["artifacts"]["test"]
    test_data_path = os.path.join(artifacts_dir , split_data_folder ,test_data_file)
    test_data = pd.read_csv(test_data_path)
    test_x = test_data.drop("quality" , axis = 1)
    test_y = test_data["quality"]

    
    model_dir = config["artifacts"]["model_data_dir"]
    model_file = config["artifacts"]["elastic_file"]
    model_path = os.path.join(artifacts_dir ,model_dir , model_file)
    lr = joblib.load(model_path) 
    predicted_value = lr.predict(test_x)
    rmse, mae, r2 = evaluate_metrics(test_y, predicted_value)

    scores_dir = config["artifacts"]["reports_dir"]
    scores_filename = config["artifacts"]["lr_scores"]

    scores_dir_path = os.path.join(artifacts_dir, scores_dir)
    create_directory([scores_dir_path])

    scores_filepath = os.path.join(scores_dir_path, scores_filename)

    scores = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }
    save_reports(report=scores, report_path=scores_filepath)
   


    ## Random Forest
    

    model_file = config["artifacts"]["random_forest"]
    model_path = os.path.join(artifacts_dir ,model_dir , model_file)
    rf = joblib.load(model_path)
    predicted_value = rf.predict(test_x)
    rmse, mae, r2 = evaluate_metrics(test_y, predicted_value)
    scores_filename = config["artifacts"]["rf_scores"]
    scores_filepath = os.path.join(scores_dir_path, scores_filename)

    scores = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }
    save_reports(report=scores, report_path=scores_filepath)

    
    




if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config" , "-c" , default= "config/config.yaml")
    # args.add_argument("--params" , "-p" , default= "params.yaml")

    parsed_args = args.parse_args()
    eval(config_path=parsed_args.config)