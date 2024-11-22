# modules/class_imbalance.py

import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_class_weights(y_train):
    """
    Computes class weights to handle class imbalance.

    Args:
        y_train (np.ndarray): Training labels.

    Returns:
        dict: Class weights.
    """
    try:
        classes, counts = np.unique(y_train, return_counts=True)
        total_samples = len(y_train)
        class_weights = {int(cls): total_samples / (len(classes) * count) for cls, count in zip(classes, counts)}
        logger.info(f"Class weights computed: {class_weights}")
        return class_weights
    except Exception as e:
        logger.error(f"An error occurred while computing class weights: {e}")
        logger.exception(e)
        raise
