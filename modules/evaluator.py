# modules/evaluator.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import logging

logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test, config):
    """
    Evaluates the trained model on the test set.

    Args:
        model: Trained Keras model.
        X_test: Test features.
        y_test: Test labels.
        config: Configuration dictionary.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    try:
        y_pred_prob = model.predict(X_test).ravel()
        y_pred = (y_pred_prob >= 0.5).astype(int)
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['precision'] = precision_score(y_test, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_test, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_test, y_pred, zero_division=0)
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_prob)
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        logger.info(f"Evaluation Metrics: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"An error occurred during model evaluation: {e}")
        logger.exception(e)
        raise