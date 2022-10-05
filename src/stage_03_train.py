from ast import arg
from src.utils.all_utils import read_yaml , create_directory
import argparse
import pandas as pd
import os
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
import joblib

def train(config_path , params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    
    ## Read the Train Data
    artifacts_dir = config["artifacts"]['artifacts_dir']
    split_data_folder = config["artifacts"]["split_data_dir"]  
    train_data_file = config["artifacts"]["train"]
    train_data_path = os.path.join(artifacts_dir , split_data_folder ,train_data_file)
    train_data = pd.read_csv(train_data_path)
    train_x = train_data.drop("quality" , axis = 1)
    train_y = train_data["quality"]

    ## Elastic Model
    alpha_par = params["model_params"]["ElasticNet"]["alpha"]
    l1_par = params["model_params"]["ElasticNet"]["l1_ratio"]
    state = params["model_params"]["ElasticNet"]["random_state"]
    lr = ElasticNet(alpha=alpha_par , l1_ratio=l1_par , random_state=state)
    lr.fit(train_x , train_y)

    ## Save the Model
    model_dir = config["artifacts"]["model_data_dir"]
    model_file = config["artifacts"]["elastic_file"]
    model_dir = os.path.join(artifacts_dir , model_dir)
    create_directory([model_dir])
    model_path = os.path.join(model_dir , model_file)
    joblib.dump(lr , model_path)

    ## Random Forest
    oob = params["model_params"]["RandomForestRegressor"]["oob_score"]
    rf = RandomForestRegressor(oob_score=oob)
    rf.fit(train_x , train_y)
    print(f"Our OOB Score is {rf.oob_score_}")

    ## Save the Model
    model_dir = config["artifacts"]["model_data_dir"]
    model_file = config["artifacts"]["random_forest"]
    model_dir = os.path.join(artifacts_dir , model_dir)
    create_directory([model_dir])
    model_path = os.path.join(model_dir , model_file)
    joblib.dump(lr , model_path)

    print("Model Saved Successfully")

    
    




if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config" , "-c" , default= "config/config.yaml")
    args.add_argument("--params" , "-p" , default= "params.yaml")

    parsed_args = args.parse_args()
    train(config_path=parsed_args.config  , params_path=parsed_args.params )