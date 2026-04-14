# Imports
import logging
import pandas as pd
import torch
import os

import data.preprocessing as preprocessing
from config import config, create_logger

logger: logging.Logger = create_logger(__name__)

# Method to run the data preprocessing steps with the number of clients from the config class
def run_preprocessing():

    # Create the output directory if it doesn't exist
    if not os.path.exists(config.PROCESSED_DATA_PATH):
        os.makedirs(config.PROCESSED_DATA_PATH)

    # Create train, validation and test directories inside the processed data directory if they don't exist
    for split in ['train', 'val', 'test']:
        split_path: str = f"{config.PROCESSED_DATA_PATH}/{split}"
        if not os.path.exists(split_path):
            os.makedirs(split_path)

    data: pd.DataFrame = preprocessing.read_csv()
    for building_id in range(1, config.NUMBER_CLIENTS + 1):

        logger.info(f'Processing building_id: {building_id}')

        df: pd.DataFrame = preprocessing.preprocess_data(data, building_id)
        df: pd.DataFrame = preprocessing.preprocess_date(df)
        df: pd.DataFrame = preprocessing.compute_net_consumption(df)
        df: pd.DataFrame = preprocessing.reorder_columns(df)
        df: pd.DataFrame = preprocessing.normalize_energy_data(df)
        df: pd.DataFrame = preprocessing.normalize_weather_data(df)

        traing_df, val_df, test_df = preprocessing.split_data(df)
        # Save the processed data for each split
        for split_name, split_df in zip(['train', 'val', 'test'], [traing_df, val_df, test_df]):
            tensor: torch.Tensor = preprocessing.df_to_tensor(split_df)
            output_file_name: str = f'{split_name}/building_{building_id}'
            preprocessing.save_tensor(tensor, output_file_name)

            logger.info(f'Saved processed {split_name} data for building_id: {building_id} to {output_file_name}')