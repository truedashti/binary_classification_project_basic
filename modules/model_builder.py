# modules/model_builder.py

import logging
import tensorflow as tf

logger = logging.getLogger(__name__)



def build_model(input_dim, model_config, hyperparameters=None):
    """
    Builds and returns a compiled Keras model.

    Args:
        input_dim (int): Input dimension of the model.
        model_config (dict): Configuration for the model architecture.
        hyperparameters (dict): Best hyperparameters obtained from tuning.

    Returns:
        model: Compiled Keras model.
    """
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Input, Dense, Dropout

        model = Sequential()
        layers = model_config['layers']

        # Add Input layer
        model.add(Input(shape=(input_dim,)))

        for i, layer in enumerate(layers[:-1]):  # Exclude the output layer for now
            units = hyperparameters.get(f'units_{i}', layer['units'])
            activation = layer['activation']
            dropout_rate = hyperparameters.get(f'dropout_{i}', layer.get('dropout', 0))

            model.add(Dense(units=units, activation=activation))
            model.add(Dropout(rate=dropout_rate))

        # Output layer
        output_layer = layers[-1]
        model.add(Dense(units=output_layer['units'], activation=output_layer['activation']))

        # Compile the model with the best optimizer and learning rate
        optimizer_name = hyperparameters.get('optimizer', model_config['optimizer'])
        learning_rate = hyperparameters.get('learning_rate', model_config.get('learning_rate', 0.001))
        if optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        model.compile(
            optimizer=optimizer,
            loss=model_config['loss'],
            metrics=model_config['metrics']
        )
        return model
    except Exception as e:
        logger.error(f"An error occurred while building the model: {e}")
        logger.exception(e)
        raise