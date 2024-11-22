import os
import logging
import logging.config
import sys
import yaml
import time
import numpy as np
import pandas as pd
import tensorflow as tf

# Import custom modules
from modules.data_loader import load_data
from modules.exploratory import explore_data
from modules.missing_values import handle_missing_values
from modules.class_imbalance import compute_class_weights
from modules.preprocessing import split_data, scale_features
from modules.model_builder import build_model
from modules.trainer import train_model
from modules.evaluator import evaluate_model
from modules.plotter import (
    plot_confusion_matrix,
    plot_precision_recall_curve_plot,
    plot_training_history,
    plot_classification_report_heatmap,
    plot_validation_f1_score,
    plot_correlation_heatmap,
    plot_top_correlation_heatmap,
    plot_shap_summary,
    plot_shap_force,
    plot_shap_waterfall,
    plot_roc_curve_plot,
    plot_hyperparameter_performance
)
from modules.interpretability import explain_model_predictions
from modules.report_generator import generate_report
from modules.hyperparameter_tuner import tune_hyperparameters

def ensure_directories():
    """
    Ensures that necessary directories exist. Creates them if they don't.
    """
    directories = ['images', 'reports', 'logs', 'hyperband_logs', 'results']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def validate_config(config):
    """
    Validates the configuration to ensure all necessary keys are present.
    """
    required_keys = {
        'plotting': ['figure_sizes', 'dpi'],
        'model': ['layers', 'loss', 'optimizer', 'metrics', 'epochs', 'batch_size'],
        'preprocessing': ['test_size', 'missing_values_strategy'],
        'report': ['executive_summary'],
        'data': ['path'],
        'logging': ['version', 'disable_existing_loggers', 'formatters', 'handlers', 'loggers'],
        'hyperparameters': []
    }

    for section, keys in required_keys.items():
        if section not in config:
            raise KeyError(f"Missing '{section}' section in configuration.")
        for key in keys:
            if key not in config[section]:
                raise KeyError(f"Missing '{key}' in configuration section '{section}'.")

    # Further validate 'plotting' -> 'figure_sizes'
    required_fig_sizes = [
        'correlation_heatmap',
        'target_distribution',
        'confusion_matrix',
        'classification_report',
        'precision_recall_curve',
        'training_history',
        'validation_f1_score',
        'shap_summary',
        'shap_force_plot',
        'shap_waterfall_plot',
        'roc_curve',
        'top_correlation_heatmap',
        'hyperparameter_performance'
    ]
    for fig in required_fig_sizes:
        if fig not in config['plotting']['figure_sizes']:
            raise KeyError(f"Missing '{fig}' in 'plotting.figure_sizes' configuration.")

def setup_gpu(logger):
    """
    Configures TensorFlow to use the GPU and enables memory growth.

    Args:
        logger (logging.Logger): Logger instance.
    """
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            tf.config.set_visible_devices(physical_devices, 'GPU')
            logger.info(f"Using GPU: {physical_devices}")
        except Exception as e:
            logger.error(f"Error setting up GPU: {e}")
    else:
        logger.warning("No GPU found. Using CPU.")

def main():
    # Load configuration
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        print(f"Configuration file '{config_path}' not found.")
        sys.exit(1)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Validate configuration
    try:
        validate_config(config)
    except KeyError as ke:
        print(f"Configuration validation error: {ke}")
        sys.exit(1)

    # Set up logging
    logging.config.dictConfig(config['logging'])
    logger = logging.getLogger(__name__)
    logger.info("Starting the Binary Classification Project...")

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    start_time = time.time()

    try:
        # Ensure necessary directories exist
        ensure_directories()

        # Configure TensorFlow to use the GPU
        setup_gpu(logger)

        # Step 1: Load Data
        step_start = time.time()
        logger.info("Loading data...")
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config['data']['path'])
        df = load_data(data_path)
        logger.info(f"Data loaded in {time.time() - step_start:.2f} seconds.")

        # Step 2: Exploratory Data Analysis
        step_start = time.time()
        logger.info("Performing Exploratory Data Analysis...")
        explore_data(df, config)
        logger.info(f"EDA completed in {time.time() - step_start:.2f} seconds.")

        # Step 3: Handle Missing Values
        step_start = time.time()
        logger.info("Handling missing values...")
        df = handle_missing_values(df, strategy=config['preprocessing'].get('missing_values_strategy', 'drop'))
        logger.info(f"Missing values handled in {time.time() - step_start:.2f} seconds.")

        # Step 4: Data Preprocessing (Splitting and Scaling)
        step_start = time.time()
        logger.info("Preprocessing data (splitting and scaling)...")
        X_train, X_test, y_train, y_test = split_data(df, test_size=config['preprocessing'].get('test_size', 0.2))

        # Check for data leakage
        overlap = X_train.index.intersection(X_test.index)
        if not overlap.empty:
            logger.error("Data leakage detected: Training and test sets have overlapping indices.")
            sys.exit(1)
        else:
            logger.info("No data leakage detected: Training and test sets are properly separated.")

        X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
        logger.info(f"Data preprocessing completed in {time.time() - step_start:.2f} seconds.")

        # Convert scaled numpy arrays back to DataFrames with original column names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

        # Convert y_train and y_test to numpy arrays of type int
        y_train = y_train.to_numpy().astype(int)
        y_test = y_test.to_numpy().astype(int)

        # Log data shapes and distributions
        logger.info(f"X_train_scaled shape: {X_train_scaled.shape}, type: {type(X_train_scaled)}")
        logger.info(f"X_test_scaled shape: {X_test_scaled.shape}, type: {type(X_test_scaled)}")
        logger.info(f"y_train shape: {y_train.shape}, unique values: {np.unique(y_train, return_counts=True)}")
        logger.info(f"y_test shape: {y_test.shape}, unique values: {np.unique(y_test, return_counts=True)}")

        # Step 5: Handle Class Imbalance
        step_start = time.time()
        logger.info("Handling class imbalance...")
        class_weights_dict = compute_class_weights(y_train)
        logger.info(f"Computed class weights: {class_weights_dict}")
        logger.info(f"Class imbalance handled in {time.time() - step_start:.2f} seconds.")

        # Step 6: Hyperparameter Tuning
        step_start = time.time()
        logger.info("Starting Hyperparameter Tuning...")
        tuner = tune_hyperparameters(
            X_train_scaled,
            y_train,
            X_val=X_test_scaled,
            y_val=y_test,
            class_weight=class_weights_dict,
            config=config
        )
        logger.info(f"Hyperparameter Tuning completed in {time.time() - step_start:.2f} seconds.")

        # Step 7: Generate Hyperparameter Performance Plots
        step_start = time.time()
        logger.info("Generating Hyperparameter Performance Plots...")
        # Generate hyperparameter performance plots
        hyperparameter_performance_paths = plot_hyperparameter_performance(tuner, config)
        logger.info(f"Hyperparameter Performance Plots generated.")

        # Get the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        logger.info(f"Best hyperparameters: {best_hps.values}")

        # Save all hyperparameter combinations and their metrics
        all_trials = []
        for trial in tuner.oracle.trials.values():
            trial_info = trial.hyperparameters.values.copy()
            trial_info['score'] = trial.score
            all_trials.append(trial_info)
        results_df = pd.DataFrame(all_trials)
        results_csv_path = os.path.join('results', 'hyperparameter_results.csv')
        results_df.to_csv(results_csv_path, index=False)
        logger.info(f"All hyperparameter combinations and their scores saved to {results_csv_path}")

        # Also save results for hyperparameter performance plotting
        results_performance_csv_path = os.path.join('results', 'hyperparameter_performance.csv')
        results_df.to_csv(results_performance_csv_path, index=False)
        logger.info(f"Hyperparameter performance data saved at {results_performance_csv_path}.")

        # Step 8: Build and Train the Model with Optimal Hyperparameters
        step_start = time.time()
        logger.info("Building and training the model with optimal hyperparameters...")
        input_dim = X_train_scaled.shape[1]
        model = build_model(input_dim, config['model'], hyperparameters=best_hps.values)
        training_result = train_model(
            model,
            X_train_scaled,
            y_train,
            X_val=X_test_scaled,
            y_val=y_test,
            class_weight=class_weights_dict,
            config=config['model']
        )
        if training_result is None:
            logger.error("train_model returned None. Expected (history, f1_callback).")
            raise ValueError("train_model returned None instead of (history, f1_callback).")
        history, f1_callback = training_result
        logger.info(f"Model trained in {time.time() - step_start:.2f} seconds with optimal hyperparameters.")

        # Step 9: Evaluate the Model
        step_start = time.time()
        logger.info("Evaluating the Model...")
        metrics = evaluate_model(model, X_test_scaled, y_test, config)
        logger.info(f"Model evaluated in {time.time() - step_start:.2f} seconds.")

        # Log the metrics to verify their values
        logger.info(f"Evaluation Metrics: {metrics}")

        # Log label distribution in y_train and y_test
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        logger.info(f"y_train distribution: {dict(zip(unique_train, counts_train))}")
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        logger.info(f"y_test distribution: {dict(zip(unique_test, counts_test))}")

        # Predict on test set to get y_pred_prob and y_pred
        y_pred_prob = model.predict(X_test_scaled).ravel()
        y_pred = (y_pred_prob >= 0.5).astype(int)
        metrics['y_pred_prob'] = y_pred_prob
        metrics['y_pred'] = y_pred
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        logger.info(f"y_pred distribution: {dict(zip(unique_pred, counts_pred))}")

        # Additional Diagnostic: Check a few sample predictions
        sample_indices = np.random.choice(len(y_test), size=10, replace=False)
        for idx in sample_indices:
            logger.info(f"Sample {idx}: y_test={y_test[idx]}, y_pred_prob={y_pred_prob[idx]:.4f}, y_pred={y_pred[idx]}")

        # Step 10: Plot Validation F1 Score Over Epochs
        step_start = time.time()
        logger.info("Plotting Validation F1 Score Over Epochs...")
        f1_path = plot_validation_f1_score(f1_callback.f1_scores, config)
        metrics['validation_f1_score_path'] = f1_path  # Save path for reporting
        logger.info(f"Validation F1 Score plotted in {time.time() - step_start:.2f} seconds.")

        # Step 11: Plotting Metrics
        logger.info("Generating Plots...")
        # Plot Correlation Heatmap
        step_start = time.time()
        logger.info("Plotting Correlation Heatmap...")
        corr_path = plot_correlation_heatmap(df, config)
        metrics['correlation_heatmap_path'] = corr_path
        logger.info(f"Correlation Heatmap plotted in {time.time() - step_start:.2f} seconds.")

        # Plot Top N Correlated Features Heatmap
        step_start = time.time()
        logger.info("Plotting Top N Correlated Features Heatmap...")
        top_corr_path = plot_top_correlation_heatmap(df, top_n=10, config=config)
        metrics['top_correlation_heatmap_path'] = top_corr_path
        logger.info(f"Top N Correlated Features Heatmap plotted in {time.time() - step_start:.2f} seconds.")

        # Plot Classification Report Heatmap
        step_start = time.time()
        logger.info("Plotting Classification Report Heatmap...")
        cr_path = plot_classification_report_heatmap(metrics['classification_report'], config)
        metrics['classification_report_path'] = cr_path
        logger.info(f"Classification Report Heatmap plotted in {time.time() - step_start:.2f} seconds.")

        # Plot Confusion Matrix
        step_start = time.time()
        logger.info("Plotting Confusion Matrix...")
        cm_path = plot_confusion_matrix(metrics['confusion_matrix'], config)
        metrics['confusion_matrix_path'] = cm_path
        logger.info(f"Confusion Matrix plotted in {time.time() - step_start:.2f} seconds.")

        # Plot Precision-Recall Curve
        step_start = time.time()
        logger.info("Plotting Precision-Recall Curve...")
        pr_path = plot_precision_recall_curve_plot(y_test, y_pred_prob, config)
        metrics['precision_recall_curve_path'] = pr_path
        logger.info(f"Precision-Recall Curve plotted in {time.time() - step_start:.2f} seconds.")

        # Plot ROC Curve
        step_start = time.time()
        logger.info("Plotting ROC Curve...")
        roc_path = plot_roc_curve_plot(y_test, y_pred_prob, config)
        metrics['roc_curve_path'] = roc_path
        logger.info(f"ROC Curve plotted in {time.time() - step_start:.2f} seconds.")

        # Plot Training History
        step_start = time.time()
        logger.info("Plotting Training History...")
        training_history_result = plot_training_history(history, config)
        if training_history_result is None:
            logger.error("plot_training_history returned None. Expected (path_loss, path_acc).")
            raise ValueError("plot_training_history returned None instead of (path_loss, path_acc).")

        path_loss, path_acc = training_history_result
        metrics['training_loss_path'] = path_loss
        metrics['training_accuracy_path'] = path_acc
        logger.info(f"Training History plotted in {time.time() - step_start:.2f} seconds.")

        # Step 12: SHAP Analysis
        step_start = time.time()
        logger.info("Performing SHAP Analysis...")

        try:
            # Define subsets for SHAP
            shap_sample_size_train = min(100, len(X_train_scaled))
            shap_sample_size_test = min(100, len(X_test_scaled))

            # For Training Set
            X_shap_train = X_train_scaled.sample(n=shap_sample_size_train, random_state=42) if shap_sample_size_train < len(X_train_scaled) else X_train_scaled.copy()

            # For Test Set
            X_shap_test = X_test_scaled.sample(n=shap_sample_size_test, random_state=42) if shap_sample_size_test < len(X_test_scaled) else X_test_scaled.copy()

            # Compute SHAP values for Training Set
            explainer_train, shap_values_train = explain_model_predictions(model, X_shap_train, config)
            logger.info(f"SHAP explanations for Training Set generated successfully.")

            # Compute SHAP values for Test Set
            explainer_test, shap_values_test = explain_model_predictions(model, X_shap_test, config)
            logger.info(f"SHAP explanations for Test Set generated successfully.")

            # Compute feature importance for Training Set
            feature_importance_train = np.abs(shap_values_train.values).mean(axis=0)
            top_feature_indices_train = np.argsort(feature_importance_train)[::-1][:5]
            top_features_train = [X_shap_train.columns[i] for i in top_feature_indices_train]
            logger.info(f"Top 5 features based on SHAP values for Training Set: {top_features_train}")

            # Compute feature importance for Test Set
            feature_importance_test = np.abs(shap_values_test.values).mean(axis=0)
            top_feature_indices_test = np.argsort(feature_importance_test)[::-1][:5]
            top_features_test = [X_shap_test.columns[i] for i in top_feature_indices_test]
            logger.info(f"Top 5 features based on SHAP values for Test Set: {top_features_test}")

            # Generate SHAP Summary Plots
            shap_summary_train_path = plot_shap_summary(shap_values_train, X_shap_train, config, title="SHAP Summary Plot - Training Set")
            metrics['shap_summary_train_path'] = shap_summary_train_path

            shap_summary_test_path = plot_shap_summary(shap_values_test, X_shap_test, config, title="SHAP Summary Plot - Test Set")
            metrics['shap_summary_test_path'] = shap_summary_test_path

            # Generate SHAP Force and Waterfall Plots for selected instances in Test Set
            selected_indices = [0, 1, 2]
            shap_plot_paths = []
            for idx in selected_indices:
                if idx < len(shap_values_test):
                    instance_name = str(X_shap_test.index[idx])
                    shap_plot_paths.append(plot_shap_force(shap_values_test, X_shap_test, idx, config, instance_name))
                    shap_plot_paths.append(plot_shap_waterfall(shap_values_test, X_shap_test, idx, config, instance_name))
                    metrics['shap_plot_paths'] = shap_plot_paths

            logger.info(f"SHAP Analysis completed in {time.time() - step_start:.2f} seconds.")

        except Exception as e:
            logger.error(f"An error occurred during SHAP Analysis: {e}")
            logger.exception(e)
            sys.exit(1)

        # Step 13: Generate PDF Report with Best Hyperparameters
        step_start = time.time()
        logger.info("Generating PDF Report with Best Hyperparameters...")
        generate_report(
            metrics=metrics,
            class_weights=class_weights_dict,
            history=history,
            f1_callback=f1_callback,
            config=config,
            shap_explainer_train=explainer_train,
            shap_values_train=shap_values_train,
            X_train_shap=X_shap_train,
            X_test_shap=X_shap_test,
            shap_plot_paths_train=[],
            shap_plot_paths_test=shap_plot_paths,
            top_features_test=top_features_test,
            top_features_train=top_features_train,
            best_hyperparameters=best_hps.values,
            hyperparameter_performance_paths=hyperparameter_performance_paths,
            report_name='Binary_Classification_Report_Best_Hyperparameters.pdf'
        )
        logger.info(f"PDF Report generated in {time.time() - step_start:.2f} seconds.")

        # Step 14: Build and Train the Model with Default Hyperparameters
        logger.info("Training model with default hyperparameters...")

        # Define default hyperparameters (adjust as needed)
        default_hyperparameters = {
            'units_0': 64,
            'units_1': 32,
            'dropout_0': 0.2,
            'dropout_1': 0.2,
            'optimizer': 'adam',
            'learning_rate': 0.001
        }

        # Build and train the model with default hyperparameters
        model_default = build_model(input_dim, config['model'], hyperparameters=default_hyperparameters)
        training_result_default = train_model(
            model_default,
            X_train_scaled,
            y_train,
            X_val=X_test_scaled,
            y_val=y_test,
            class_weight=class_weights_dict,
            config=config['model']
        )
        if training_result_default is None:
            logger.error("train_model returned None. Expected (history, f1_callback).")
            raise ValueError("train_model returned None instead of (history, f1_callback).")
        history_default, f1_callback_default = training_result_default
        logger.info("Model trained with default hyperparameters.")

        # Evaluate the model with default hyperparameters
        metrics_default = evaluate_model(model_default, X_test_scaled, y_test, config)
        logger.info(f"Evaluation Metrics with Default Hyperparameters: {metrics_default}")

        # Predict on test set to get y_pred_prob and y_pred
        y_pred_prob_default = model_default.predict(X_test_scaled).ravel()
        y_pred_default = (y_pred_prob_default >= 0.5).astype(int)
        metrics_default['y_pred_prob'] = y_pred_prob_default
        metrics_default['y_pred'] = y_pred_default

        # Generate the general report with default hyperparameters
        step_start = time.time()
        logger.info("Generating PDF Report with Default Hyperparameters...")

        # Ensure that metrics_default contains paths to the images
        # Generate plots for the default model
        logger.info("Generating plots for the default model...")

        # Plot Validation F1 Score
        f1_path_default = plot_validation_f1_score(f1_callback_default.f1_scores, config)
        metrics_default['validation_f1_score_path'] = f1_path_default

        # Plot Classification Report Heatmap
        cr_path_default = plot_classification_report_heatmap(metrics_default['classification_report'], config)
        metrics_default['classification_report_path'] = cr_path_default

        # Plot Confusion Matrix
        cm_path_default = plot_confusion_matrix(metrics_default['confusion_matrix'], config)
        metrics_default['confusion_matrix_path'] = cm_path_default

        # Plot Precision-Recall Curve
        pr_path_default = plot_precision_recall_curve_plot(y_test, y_pred_prob_default, config)
        metrics_default['precision_recall_curve_path'] = pr_path_default

        # Plot ROC Curve
        roc_path_default = plot_roc_curve_plot(y_test, y_pred_prob_default, config)
        metrics_default['roc_curve_path'] = roc_path_default

        # Plot Training History
        training_history_result_default = plot_training_history(history_default, config)
        if training_history_result_default is None:
            logger.error("plot_training_history returned None for default model.")
            raise ValueError("plot_training_history returned None instead of (path_loss, path_acc).")

        path_loss_default, path_acc_default = training_history_result_default
        metrics_default['training_loss_path'] = path_loss_default
        metrics_default['training_accuracy_path'] = path_acc_default

        # Perform SHAP Analysis for Default Model
        try:
            # Define subsets for SHAP
            shap_sample_size_train = min(100, len(X_train_scaled))
            shap_sample_size_test = min(100, len(X_test_scaled))

            # For Training Set
            X_shap_train_default = X_train_scaled.sample(n=shap_sample_size_train, random_state=42) if shap_sample_size_train < len(X_train_scaled) else X_train_scaled.copy()

            # For Test Set
            X_shap_test_default = X_test_scaled.sample(n=shap_sample_size_test, random_state=42) if shap_sample_size_test < len(X_test_scaled) else X_test_scaled.copy()

            # Compute SHAP values for Training Set
            explainer_train_default, shap_values_train_default = explain_model_predictions(model_default, X_shap_train_default, config)
            logger.info(f"SHAP explanations for Training Set (Default Model) generated successfully.")

            # Compute SHAP values for Test Set
            explainer_test_default, shap_values_test_default = explain_model_predictions(model_default, X_shap_test_default, config)
            logger.info(f"SHAP explanations for Test Set (Default Model) generated successfully.")

            # Compute feature importance for Training Set
            feature_importance_train_default = np.abs(shap_values_train_default.values).mean(axis=0)
            top_feature_indices_train_default = np.argsort(feature_importance_train_default)[::-1][:5]
            top_features_train_default = [X_shap_train_default.columns[i] for i in top_feature_indices_train_default]
            logger.info(f"Top 5 features based on SHAP values for Training Set (Default Model): {top_features_train_default}")

            # Compute feature importance for Test Set
            feature_importance_test_default = np.abs(shap_values_test_default.values).mean(axis=0)
            top_feature_indices_test_default = np.argsort(feature_importance_test_default)[::-1][:5]
            top_features_test_default = [X_shap_test_default.columns[i] for i in top_feature_indices_test_default]
            logger.info(f"Top 5 features based on SHAP values for Test Set (Default Model): {top_features_test_default}")

            # Generate SHAP Summary Plots
            shap_summary_train_path_default = plot_shap_summary(shap_values_train_default, X_shap_train_default, config, title="SHAP Summary Plot - Training Set")
            metrics_default['shap_summary_train_path'] = shap_summary_train_path_default

            shap_summary_test_path_default = plot_shap_summary(shap_values_test_default, X_shap_test_default, config, title="SHAP Summary Plot - Test Set")
            metrics_default['shap_summary_test_path'] = shap_summary_test_path_default

            # Generate SHAP Force and Waterfall Plots for selected instances in Test Set
            selected_indices = [0, 1, 2]
            shap_plot_paths_default = []
            for idx in selected_indices:
                if idx < len(shap_values_test_default):
                    instance_name = str(X_shap_test_default.index[idx])
                    shap_plot_paths_default.append(plot_shap_force(shap_values_test_default, X_shap_test_default, idx, config, instance_name))
                    shap_plot_paths_default.append(plot_shap_waterfall(shap_values_test_default, X_shap_test_default, idx, config, instance_name))
                    metrics_default['shap_plot_paths'] = shap_plot_paths_default

            logger.info(f"SHAP Analysis for Default Model completed successfully.")

        except Exception as e:
            logger.error(f"An error occurred during SHAP Analysis for Default Model: {e}")
            logger.exception(e)
            sys.exit(1)

        generate_report(
            metrics=metrics_default,
            class_weights=class_weights_dict,
            history=history_default,
            f1_callback=f1_callback_default,
            config=config,
            shap_explainer_train=explainer_train_default,
            shap_values_train=shap_values_train_default,
            X_train_shap=X_shap_train_default,
            X_test_shap=X_shap_test_default,
            shap_plot_paths_train=[],
            shap_plot_paths_test=shap_plot_paths_default,
            top_features_test=top_features_test_default,
            top_features_train=top_features_train_default,
            best_hyperparameters=default_hyperparameters,
            hyperparameter_performance_paths=hyperparameter_performance_paths,
            report_name='Binary_Classification_Report_Default_Hyperparameters.pdf'
        )
        logger.info(f"PDF Report generated in {time.time() - step_start:.2f} seconds.")

        total_time = time.time() - start_time
        logger.info(f"All tasks completed successfully in {total_time:.2f} seconds!")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.exception(e)
        sys.exit(1)

if __name__ == "__main__":
    main()
