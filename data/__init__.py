# Imports
import logging
import pandas as pd
import torch

import data.preprocessing as preprocessing
from config import config, create_logger

logger: logging.Logger = create_logger(__name__)

# Method to run the data preprocessing steps with the number of clients from the config class
def run_preprocessing():

    data: pd.DataFrame = preprocessing.read_csv()
    for building_id in range(1, config.NUMBER_CLIENTS + 1):

        logger.info(f'Processing building_id: {building_id}')

        df: pd.DataFrame = preprocessing.preprocess_data(data, building_id)
        df: pd.DataFrame = preprocessing.preprocess_date(df)
        df: pd.DataFrame = preprocessing.compute_net_consumption(df)
        df: pd.DataFrame = preprocessing.reorder_columns(df)
        tensor: torch.Tensor = preprocessing.df_to_tensor(df)
        output_file_name: str = f'building_{building_id}'
        preprocessing.save_tensor(tensor, output_file_name)

        logger.info(f'Saved processed data for building_id: {building_id} to {output_file_name}')