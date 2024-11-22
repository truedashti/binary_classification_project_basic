# modules/missing_values.py

import pandas as pd
import logging

logger = logging.getLogger(__name__)

def handle_missing_values(df, strategy='drop'):
    """
    Handles missing values in the DataFrame.

    Args:
        df (pd.DataFrame): The dataset.
        strategy (str): Strategy to handle missing values. Options: 'drop', 'mean', 'median', 'mode'.

    Returns:
        pd.DataFrame: DataFrame after handling missing values.
    """
    try:
        if strategy == 'drop':
            initial_shape = df.shape
            df = df.dropna()
            logger.info(f"Dropped missing values. Shape changed from {initial_shape} to {df.shape}.")
        elif strategy in ['mean', 'median', 'mode']:
            if strategy == 'mode':
                df = df.fillna(df.mode().iloc[0])
            elif strategy == 'mean':
                df = df.fillna(df.mean())
            elif strategy == 'median':
                df = df.fillna(df.median())
            logger.info(f"Filled missing values using {strategy} strategy.")
        else:
            logger.error(f"Unsupported missing values strategy: {strategy}")
            raise ValueError(f"Unsupported missing values strategy: {strategy}")
        return df
    except Exception as e:
        logger.error(f"An error occurred while handling missing values: {e}")
        logger.exception(e)
        raise
