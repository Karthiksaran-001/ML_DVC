from ast import arg
from src.utils.all_utils import read_yaml
import argparse
import pandas as pd





if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config" "-c" , default= "config/config.yaml")

    parsed_args = args.parse_args()