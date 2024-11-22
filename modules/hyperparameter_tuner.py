import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras_tuner import HyperModel, Hyperband
import logging
import os
import shutil

logger = logging.getLogger(__name__)

class BinaryClassificationHyperModel(HyperModel):
    def __init__(self, input_dim, config):
        self.input_dim = input_dim
        self.config = config

    def build(self, hp):
        model = Sequential()
        # Hyperparameter tuning for number of units and dropout rate
        num_layers = len(self.config['model']['layers']) - 1  # Exclude output layer
        for i in range(num_layers):
            units = hp.Int('units_' + str(i),
                           min_value=self.config['hyperparameters']['units']['min'],
                           max_value=self.config['hyperparameters']['units']['max'],
                           step=self.config['hyperparameters']['units']['step'])
            activation = self.config['model']['layers'][i]['activation']
            # Use hp.Choice for dropout to ensure specific values are used
            min_dropout = self.config['hyperparameters']['dropout']['min']
            max_dropout = self.config['hyperparameters']['dropout']['max']
            dropout_step = self.config['hyperparameters']['dropout']['step']
            num_dropout_values = int((max_dropout - min_dropout) / dropout_step) + 1
            dropout_choices = [round(min_dropout + i * dropout_step, 1) for i in range(num_dropout_values)]
            dropout_rate = hp.Choice('dropout_' + str(i), values=dropout_choices)
            if i == 0:
                model.add(Dense(units=units, activation=activation, input_dim=self.input_dim))
            else:
                model.add(Dense(units=units, activation=activation))
            model.add(Dropout(rate=dropout_rate))
        # Output layer
        output_layer = self.config['model']['layers'][-1]
        model.add(Dense(units=output_layer['units'], activation=output_layer['activation']))

        # Hyperparameter tuning for optimizer and learning rate
        optimizer_name = hp.Choice('optimizer', self.config['hyperparameters']['optimizer']['values'])
        learning_rate = hp.Choice('learning_rate', self.config['hyperparameters']['learning_rate']['values'])
        if optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        model.compile(loss=self.config['model']['loss'],
                      optimizer=optimizer,
                      metrics=self.config['model']['metrics'])
        return model

def tune_hyperparameters(X_train, y_train, X_val, y_val, class_weight, config):
    """
    Tunes hyperparameters using Keras Tuner.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        class_weight: Class weights.
        config: Configuration dictionary.

    Returns:
        tuner: The Keras Tuner object after tuning.
    """
    try:
        # Delete the hyperband_logs directory to start fresh
        tuner_dir = 'hyperband_logs'
        if os.path.exists(tuner_dir):
            shutil.rmtree(tuner_dir)
            logger.info(f"Deleted existing tuner directory: {tuner_dir}")

        input_dim = X_train.shape[1]
        hypermodel = BinaryClassificationHyperModel(input_dim, config)

        tuner = Hyperband(
            hypermodel,
            objective='val_accuracy',
            max_epochs=20,
            factor=3,
            directory=tuner_dir,
            project_name='heart_disease_classification'
        )

        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        tuner.search(X_train, y_train,
                     epochs=50,
                     validation_data=(X_val, y_val),
                     class_weight=class_weight,
                     callbacks=[stop_early],
                     verbose=1)

        logger.info("Hyperparameter tuning completed successfully.")
        return tuner
    except Exception as e:
        logger.error(f"An error occurred during hyperparameter tuning: {e}")
        logger.exception(e)
        raise
