from ast import arg
from src.utils.all_utils import read_yaml , create_directory  , save_local_df
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_save_data(config_path , params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    # save dataset in the local directory
    # create path to directory: artifacts/raw_local_dir/data.csv
    artifacts_dir = config["artifacts"]['artifacts_dir']
    raw_local_dir = config["artifacts"]['raw_local_dir']
    raw_local_file = config["artifacts"]['raw_local_file']

    raw_local_file_path = os.path.join(artifacts_dir, raw_local_dir , raw_local_file)
    df = pd.read_csv(raw_local_file_path)
    split_ratio = params["base"]["test_size"]
    random_state = params["base"]["random_state"]
    train , test = train_test_split(df , test_size=split_ratio , random_state=random_state)
    
    ## Store the Train and test Path
    split_data_folder = config["artifacts"]["split_data_dir"]
    create_directory([os.path.join(artifacts_dir  , split_data_folder)])
    train_data_file = config["artifacts"]["train"]
    test_data_file = config["artifacts"]["test"]

    train_data_path = os.path.join(artifacts_dir , split_data_folder ,train_data_file)
    test_data_path = os.path.join(artifacts_dir , split_data_folder ,test_data_file)
    
    for data , data_path in (train , train_data_path) , (test , test_data_path):
        save_local_df(data , data_path)
   




if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config" , "-c" , default= "config/config.yaml")
    args.add_argument("--params" , "-p" , default= "params.yaml")

    parsed_args = args.parse_args()
    split_save_data(config_path=parsed_args.config  , params_path=parsed_args.params )