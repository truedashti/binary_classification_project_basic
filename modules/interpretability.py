# modules/interpretability.py

import shap
import logging

logger = logging.getLogger(__name__)

def explain_model_predictions(model, X, config):
    """
    Generates SHAP explanations for the given data.

    Args:
        model (tf.keras.Model): The trained Keras model.
        X (pd.DataFrame): DataFrame for which SHAP values are to be computed.
        config (dict): Configuration dictionary.

    Returns:
        tuple: (explainer, shap_values)
    """
    try:
        # Initialize SHAP Explainer
        explainer = shap.Explainer(model.predict, X)
        shap_values = explainer(X)
        logger.info("SHAP explanations generated successfully.")
        return explainer, shap_values
    except Exception as e:
        logger.error(f"An error occurred while explaining model predictions: {e}")
        logger.exception(e)
        raise
