# modules/exploratory.py

import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

logger = logging.getLogger(__name__)

def explore_data(df, config):
    """
    Performs exploratory data analysis.

    Args:
        df (pd.DataFrame): The dataset.
        config (dict): Configuration dictionary.
    """
    try:
        # Example EDA: Distribution of target variable
        plt.figure(figsize=config['plotting']['figure_sizes']['target_distribution'], dpi=config['plotting']['dpi'])
        sns.countplot(x='target', data=df, palette='Set2')
        plt.title('Distribution of Target Variable', fontsize=14)
        plt.xlabel('Target', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.tight_layout()
        target_dist_path = os.path.join('images', 'target_distribution.png')
        plt.savefig(target_dist_path)
        plt.close()
        logger.info(f"Target variable distribution plot saved at {target_dist_path}.")

        # Add more EDA plots if needed

    except Exception as e:
        logger.error(f"An error occurred during Exploratory Data Analysis: {e}")
        logger.exception(e)
        raise
