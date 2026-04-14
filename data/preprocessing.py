# Imports
import pandas as pd
import numpy as np
import torch

from config import config

# Methods for preprocessing data
def read_csv() -> pd.DataFrame:
    """
    Reads the input CSV file and returns it as a pandas DataFrame.
    The file path is constructed using the configuration settings.

    Returns:
        pd.DataFrame: The DataFrame containing the data from the CSV file.
    """
    input_file_path: str = f"{config.INPUT_DATA_PATH}/{config.INPUT_DATA_FILENAME}"
    df: pd.DataFrame = pd.read_csv(input_file_path)
    return df

def preprocess_data(df: pd.DataFrame, building_id: int) -> pd.DataFrame:
    """
    Preprocesses the input DataFrame by selecting specific columns and returning a new DataFrame.
    The columns selected include 'date', 'load_X', 'pv_X', 'temp', 'dwpt', 'rhum', 'wdir', 'wspd', and 'pres', where X is the building_id.

    Args:
        df (pd.DataFrame): The input DataFrame to preprocess.
        building_id (int): The ID of the building to select the corresponding load and pv columns.

    Returns:
        pd.DataFrame: A new DataFrame containing only the selected columns.
    """
    columns_to_copy: list[str] = ['date', f'load_{building_id}', f'pv_{building_id}', 'temp', 'dwpt', 'rhum', 'wdir', 'wspd', 'pres']
    new_df: pd.DataFrame = df[columns_to_copy].copy()
    new_df.rename(columns = {f'load_{building_id}': 'load', f'pv_{building_id}': 'pv'}, inplace = True)
    return new_df

def preprocess_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the 'date' column in the input DataFrame by extracting the weekday, hour, and minute information.
    The method creates new columns for the weekday, sinus of the time (hour), and cosinus of the time (hour).

    Args:
        df (pd.DataFrame): The input DataFrame containing a 'date' column to preprocess.
    Returns:
        pd.DataFrame: A new DataFrame with the original columns plus the new 'weekday', 'sin_time', and 'cos_time' columns.
    """
    df['date'] = pd.to_datetime(df['date'])
    df['weekday'] = df['date'].dt.weekday
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['time_in_minutes'] = df['hour'] * 60 + df['minute']
    df['tod_sin'] = np.sin(2 * np.pi * df['time_in_minutes'] / 1440)
    df['tod_cos'] = np.cos(2 * np.pi * df['time_in_minutes'] / 1440)
    return df.drop(columns = ['date', 'hour', 'minute', 'time_in_minutes'])

def compute_net_consumption(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the net consumption for a given building by subtracting the photovoltaic (PV) generation from the load.
    The method creates a new column 'net' in the DataFrame which contains the computed values.

    Args:
        df (pd.DataFrame): The input DataFrame containing 'load' and 'pv' columns.

    Returns:
        pd.DataFrame: A new DataFrame with the 'net' column added.
    """
    df['net'] = df['load'] - df['pv']
    return df

def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorders the columns of the input DataFrame to a specific order.
    The desired order of columns is: 'weekday', 'tod_sin', 'tod_cos', 'temp', 'rhum', 'wspd', 'wdir', 'pres', 'load', 'pv', 'net'.

    Args:
        df (pd.DataFrame): The input DataFrame with columns to reorder.
    Returns:
        pd.DataFrame: A new DataFrame with columns reordered to the specified order.
    """
    desired_order: list[str] = ['weekday', 'tod_sin', 'tod_cos', 'temp', 'rhum', 'wspd', 'wdir', 'pres', 'load', 'pv', 'net']
    return df[desired_order]

def normalize_energy_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the 'load', 'pv', and 'net' columns in the input DataFrame using min-max normalization.
    The formula used for normalization is: (x - min) / (max - min), where min and max are the minimum and maximum values of the respective column.

    Args:
        df (pd.DataFrame): The input DataFrame containing 'load', 'pv', and 'net' columns to normalize.

    Returns:
        pd.DataFrame: A new DataFrame with the normalized columns.
    """
    columns_to_normalize: list[str] = ['load', 'pv', 'net']
    for col in columns_to_normalize:
        min_val: float = df[col].min()
        max_val: float = df[col].max()
        if max_val != min_val:
            df[col] = (df[col] - min_val) / (max_val - min_val)
    return df

def normalize_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the weather-related columns in the input DataFrame using standardization (z-score normalization).
    The formula used for normalization is: (x - mean) / std, where mean and std are the mean and standard deviation of the respective column.

    Args:
        df (pd.DataFrame): The input DataFrame containing weather-related columns to normalize.

    Returns:
        pd.DataFrame: A new DataFrame with the normalized weather-related columns.
    """
    weather_columns: list[str] = ['temp', 'dwpt', 'rhum', 'wdir', 'wspd', 'pres']
    for col in weather_columns:
        mean_val: float = df[col].mean()
        std_val: float = df[col].std()
        if std_val != 0:
            df[col] = (df[col] - mean_val) / std_val
    return df

def df_to_tensor(df: pd.DataFrame) -> torch.Tensor:
    """
    Converts a given pandas DataFrame to a PyTorch tensor.

    Args:
        df (pd.DataFrame): The input DataFrame to convert.

    Returns:
        torch.Tensor: A PyTorch tensor containing the data from the DataFrame.
    """
    return torch.tensor(df.values, dtype = torch.float32)

def save_tensor(tensor: torch.Tensor, name: str) -> None:
    """
    Saves a given PyTorch tensor to a file in the output directory specified in the configuration settings.
    The file is saved with the provided name and a '.pt' extension.

    Args:
        tensor (torch.Tensor): The PyTorch tensor to save.
        name (str): The name to use for the saved file (without extension).

    Returns:
        None
    """
    output_file_path: str = f"{config.PROCESSED_DATA_PATH}/{name}.pt"
    torch.save(tensor, output_file_path)
    return

def load_tensor(name: str) -> torch.Tensor:
    """
    Loads a PyTorch tensor from a file in the output directory specified in the configuration settings.
    The file is expected to have the provided name and a '.pt' extension.

    Args:
        name (str): The name of the file to load (without extension).

    Returns:
        torch.Tensor: The loaded PyTorch tensor.
    """
    input_file_path: str = f"{config.PROCESSED_DATA_PATH}/{name}.pt"
    return torch.load(input_file_path)