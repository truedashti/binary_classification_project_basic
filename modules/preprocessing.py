# modules/preprocessing.py

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# modules/preprocessing.py

def split_data(df, test_size=0.2):
    """
    Splits the data into training and testing sets.

    Args:
        df (pd.DataFrame): The dataset.
        test_size (float): Proportion of the dataset to include in the test split.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    try:
        X = df.drop('target', axis=1)
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y  # Stratify to maintain class distribution
        )
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"An error occurred while splitting data: {e}")
        logger.exception(e)
        raise


def scale_features(X_train, X_test):
    """
    Scales the features using StandardScaler.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.

    Returns:
        tuple: Scaled X_train and X_test as numpy arrays.
    """
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logger.info("Features scaled using StandardScaler.")
        return X_train_scaled, X_test_scaled
    except Exception as e:
        logger.error(f"An error occurred while scaling features: {e}")
        logger.exception(e)
        raise
