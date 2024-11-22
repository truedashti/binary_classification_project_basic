# modules/trainer.py

import logging
from tensorflow.keras.callbacks import Callback, EarlyStopping
from sklearn.metrics import f1_score
import numpy as np

logger = logging.getLogger(__name__)

class F1ScoreCallback(Callback):
    """
    Custom Keras Callback to compute F1 score at the end of each epoch.
    """
    def __init__(self, validation_data):
        super(F1ScoreCallback, self).__init__()
        self.validation_data = validation_data
        self.f1_scores = []

    def on_epoch_end(self, epoch, logs=None):
        try:
            # Predict probabilities
            y_val_pred_prob = self.model.predict(self.validation_data[0])
            # Convert probabilities to class labels
            y_val_pred = (y_val_pred_prob >= 0.5).astype(int).ravel()
            y_val_true = self.validation_data[1].ravel()
            # Calculate F1 Score using sklearn
            f1 = f1_score(y_val_true, y_val_pred, zero_division=0)
            self.f1_scores.append(f1)
            logger.info(f"Epoch {epoch+1}: Validation F1 Score = {f1:.4f}")
        except Exception as e:
            logger.error(f"An error occurred in F1ScoreCallback: {e}")
            logger.exception(e)
            raise

def train_model(model, X_train, y_train, X_val, y_val, class_weight, config):
    """
    Trains the Keras model with given data and configurations.

    Args:
        model (tf.keras.Model): The compiled Keras model.
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_val (pd.DataFrame or np.ndarray): Validation features.
        y_val (np.ndarray): Validation labels.
        class_weight (dict): Class weights to handle class imbalance.
        config (dict): Training configurations.

    Returns:
        tuple: (history, f1_callback)
    """
    try:
        # Initialize the custom F1ScoreCallback
        f1_callback = F1ScoreCallback(validation_data=(X_val, y_val))

        # Early Stopping Callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            class_weight=class_weight,
            callbacks=[f1_callback, early_stopping],
            verbose=1
        )
        logger.info("Model training completed successfully.")
        return history, f1_callback
    except Exception as e:
        logger.error(f"An error occurred during model training: {e}")
        logger.exception(e)
        raise
