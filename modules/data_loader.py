# modules/data_loader.py

import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_data(data_path):
    """
    Loads data from the specified CSV file.

    Args:
        data_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded from {data_path} with shape {df.shape}.")
        return df
    except Exception as e:
        logger.error(f"An error occurred while loading data: {e}")
        logger.exception(e)
        raise
