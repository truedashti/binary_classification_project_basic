# modules/plotter.py

import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
import shap  # Added import for shap

logger = logging.getLogger(__name__)

def plot_validation_f1_score(f1_scores, config):
    try:
        plt.figure(figsize=config['plotting']['figure_sizes']['validation_f1_score'], dpi=config['plotting']['dpi'])
        plt.plot(range(1, len(f1_scores) + 1), f1_scores, marker='o')
        plt.title('Validation F1 Score Over Epochs', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)
        plt.tight_layout()
        path = os.path.join('images', 'validation_f1_score.png')
        plt.savefig(path)
        plt.close()
        logger.info(f"Validation F1 Score plot saved at {path}.")
        return path
    except Exception as e:
        logger.error(f"An error occurred while plotting Validation F1 Score: {e}")
        logger.exception(e)
        raise

def plot_correlation_heatmap(df, config):
    try:
        plt.figure(figsize=config['plotting']['figure_sizes']['correlation_heatmap'], dpi=config['plotting']['dpi'])
        corr = df.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Heatmap', fontsize=14)
        plt.tight_layout()
        path = os.path.join('images', 'correlation_heatmap.png')
        plt.savefig(path)
        plt.close()
        logger.info(f"Correlation Heatmap saved at {path}.")
        return path
    except Exception as e:
        logger.error(f"An error occurred while plotting Correlation Heatmap: {e}")
        logger.exception(e)
        raise

def plot_top_correlation_heatmap(df, top_n, config):
    try:
        corr = df.corr()
        target_corr = corr['target'].abs().sort_values(ascending=False)
        top_features = target_corr.index[1:top_n+1]
        plt.figure(figsize=config['plotting']['figure_sizes']['top_correlation_heatmap'], dpi=config['plotting']['dpi'])
        sns.heatmap(df[top_features].corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.title(f'Top {top_n} Correlated Features Heatmap', fontsize=14)
        plt.tight_layout()
        path = os.path.join('images', 'top_correlation_heatmap.png')
        plt.savefig(path)
        plt.close()
        logger.info(f"Top {top_n} Correlated Features Heatmap saved at {path}.")
        return path
    except Exception as e:
        logger.error(f"An error occurred while plotting Top Correlation Heatmap: {e}")
        logger.exception(e)
        raise

def plot_classification_report_heatmap(classification_report_dict, config):
    try:
        cr_df = pd.DataFrame(classification_report_dict).iloc[:-1, :].T
        plt.figure(figsize=config['plotting']['figure_sizes']['classification_report'], dpi=config['plotting']['dpi'])
        sns.heatmap(cr_df, annot=True, cmap='Blues')
        plt.title('Classification Report Heatmap', fontsize=14)
        plt.tight_layout()
        path = os.path.join('images', 'classification_report_heatmap.png')
        plt.savefig(path)
        plt.close()
        logger.info(f"Classification Report Heatmap saved at {path}.")
        return path
    except Exception as e:
        logger.error(f"An error occurred while plotting Classification Report Heatmap: {e}")
        logger.exception(e)
        raise

def plot_confusion_matrix(confusion_matrix_array, config):
    try:
        plt.figure(figsize=config['plotting']['figure_sizes']['confusion_matrix'], dpi=config['plotting']['dpi'])
        sns.heatmap(confusion_matrix_array, annot=True, fmt='d', cmap='YlGnBu')
        plt.title('Confusion Matrix', fontsize=14)
        plt.ylabel('Actual Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        path = os.path.join('images', 'confusion_matrix.png')
        plt.savefig(path)
        plt.close()
        logger.info(f"Confusion Matrix saved at {path}.")
        return path
    except Exception as e:
        logger.error(f"An error occurred while plotting Confusion Matrix: {e}")
        logger.exception(e)
        raise

def plot_precision_recall_curve_plot(y_test, y_pred_prob, config):
    try:
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
        plt.figure(figsize=config['plotting']['figure_sizes']['precision_recall_curve'], dpi=config['plotting']['dpi'])
        plt.plot(recall, precision, marker='.')
        plt.title('Precision-Recall Curve', fontsize=14)
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.tight_layout()
        path = os.path.join('images', 'precision_recall_curve.png')
        plt.savefig(path)
        plt.close()
        logger.info(f"Precision-Recall Curve saved at {path}.")
        return path
    except Exception as e:
        logger.error(f"An error occurred while plotting Precision-Recall Curve: {e}")
        logger.exception(e)
        raise

def plot_roc_curve_plot(y_test, y_pred_prob, config):
    try:
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        plt.figure(figsize=config['plotting']['figure_sizes']['roc_curve'], dpi=config['plotting']['dpi'])
        plt.plot(fpr, tpr, marker='.')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.title('ROC Curve', fontsize=14)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.tight_layout()
        path = os.path.join('images', 'roc_curve.png')
        plt.savefig(path)
        plt.close()
        logger.info(f"ROC Curve saved at {path}.")
        return path
    except Exception as e:
        logger.error(f"An error occurred while plotting ROC Curve: {e}")
        logger.exception(e)
        raise

def plot_training_history(history, config):
    try:
        # Plot Training and Validation Loss
        plt.figure(figsize=config['plotting']['figure_sizes']['training_history'], dpi=config['plotting']['dpi'])
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend()
        plt.tight_layout()
        path_loss = os.path.join('images', 'training_validation_loss.png')
        plt.savefig(path_loss)
        plt.close()
        logger.info(f"Training and Validation Loss plot saved at {path_loss}.")

        # Plot Training and Validation Accuracy
        plt.figure(figsize=config['plotting']['figure_sizes']['training_history'], dpi=config['plotting']['dpi'])
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend()
        plt.tight_layout()
        path_acc = os.path.join('images', 'training_validation_accuracy.png')
        plt.savefig(path_acc)
        plt.close()
        logger.info(f"Training and Validation Accuracy plot saved at {path_acc}.")

        return path_loss, path_acc
    except Exception as e:
        logger.error(f"An error occurred while plotting Training History: {e}")
        logger.exception(e)
        raise



def plot_hyperparameter_performance(tuner, config):
    """
    Plots and saves the hyperparameter tuning performance, including pair plots and heatmaps.

    Args:
        tuner: The Keras Tuner object after tuning.
        config: Configuration dictionary.

    Returns:
        list: Paths to the saved hyperparameter performance plots.
    """
    try:
        hp_performance = []
        # Retrieve all trials
        for trial in tuner.oracle.trials.values():
            hp_values = trial.hyperparameters.values.copy()
            hp_values['Score'] = trial.score
            hp_performance.append(hp_values)

        if not hp_performance:
            logger.error("No hyperparameter tuning trials were found.")
            return []

        df = pd.DataFrame(hp_performance)

        # Ensure correct data types
        for col in df.columns:
            if col != 'Score':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Save the DataFrame to CSV
        results_csv_path = os.path.join('results', 'hyperparameter_performance.csv')
        df.to_csv(results_csv_path, index=False)
        logger.info(f"Hyperparameter performance data saved at {results_csv_path}.")

        # Pair Plot
        plot_dir = os.path.join('images', 'hyperparameter_performance')
        os.makedirs(plot_dir, exist_ok=True)

        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove 'Score' from numeric_cols if present
        if 'Score' in numeric_cols:
            numeric_cols.remove('Score')

        if len(numeric_cols) >= 2:
            sns.pairplot(df, vars=numeric_cols, diag_kind='kde')
            plt.suptitle('Hyperparameter Pair Plot', y=1.02, fontsize=16)
            pairplot_path = os.path.join(plot_dir, 'hyperparameter_pairplot.png')
            plt.savefig(pairplot_path)
            plt.close()
            logger.info(f"Hyperparameter pair plot saved at {pairplot_path}.")
        else:
            logger.warning("Not enough numeric hyperparameters for pair plot.")
            pairplot_path = None

        # Heatmap of hyperparameter correlations
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols + ['Score']].corr()
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr, annot=True, cmap='coolwarm')
            plt.title('Hyperparameter Correlation Heatmap', fontsize=16)
            heatmap_path = os.path.join(plot_dir, 'hyperparameter_correlation_heatmap.png')
            plt.savefig(heatmap_path)
            plt.close()
            logger.info(f"Hyperparameter correlation heatmap saved at {heatmap_path}.")
        else:
            logger.warning("Not enough numeric hyperparameters for correlation heatmap.")
            heatmap_path = None

        # Individual hyperparameter vs. score plots
        plot_paths = []
        if pairplot_path:
            plot_paths.append(pairplot_path)
        if heatmap_path:
            plot_paths.append(heatmap_path)

        for hp_name in df.columns.drop('Score'):
            plt.figure(figsize=config['plotting']['figure_sizes']['hyperparameter_performance'], dpi=config['plotting']['dpi'])
            if np.issubdtype(df[hp_name].dtype, np.number):
                sns.scatterplot(x=hp_name, y='Score', data=df)
                if hp_name == 'learning_rate':
                    plt.xscale('log')
            else:
                # Since the variable is categorical, use a boxplot
                sns.boxplot(x=hp_name, y='Score', data=df)
            plt.xlabel(hp_name, fontsize=12)
            plt.ylabel('Validation Score', fontsize=12)
            plt.title(f'Hyperparameter Tuning: {hp_name}', fontsize=14)
            plt.tight_layout()
            safe_hp_name = hp_name.replace('/', '_').replace('\\', '_')
            plot_path = os.path.join(plot_dir, f'{safe_hp_name}.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Hyperparameter performance plot saved at {plot_path}.")
            plot_paths.append(plot_path)

        return plot_paths
    except Exception as e:
        logger.error(f"An error occurred while plotting hyperparameter performance: {e}")
        logger.exception(e)
        raise

def plot_shap_waterfall(shap_values, X_shap, idx, config, instance_name):
    try:
        # Use the shap_values object directly for the specified index
        shap_value = shap_values[idx]

        # Limit the values to 4 decimal digits
        shap_value.values = np.round(shap_value.values, 4)
        shap_value.base_values = np.round(shap_value.base_values, 4)
        shap_value.data = np.round(shap_value.data, 4)

        # Create the waterfall plot
        plt.figure(figsize=config['plotting']['figure_sizes']['shap_waterfall_plot'], dpi=config['plotting']['dpi'])
        shap.plots.waterfall(shap_value, show=False)
        plt.title(f"SHAP Waterfall Plot for Instance {instance_name}", fontsize=14)
        path = os.path.join('images', f'shap_waterfall_plot_{instance_name}.png')
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        logger.info(f"SHAP Waterfall Plot saved at {path}.")
        return path
    except Exception as e:
        logger.error(f"An error occurred while plotting SHAP Waterfall Plot: {e}")
        logger.exception(e)
        raise

def plot_shap_force(shap_values, X_shap, idx, config, instance_name):
    try:
        # Use the shap_values object directly for the specified index
        shap_value = shap_values[idx]

        # Limit the values to 4 decimal digits
        shap_value.values = np.round(shap_value.values, 4)
        shap_value.base_values = np.round(shap_value.base_values, 4)
        shap_value.data = np.round(shap_value.data, 4)

        # Create the force plot
        plt.figure(figsize=config['plotting']['figure_sizes']['shap_force_plot'], dpi=config['plotting']['dpi'])
        shap.plots.force(shap_value, matplotlib=True, show=False)
        plt.title(f"SHAP Force Plot for Instance {instance_name}", fontsize=14)
        path = os.path.join('images', f'shap_force_plot_{instance_name}.png')
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        logger.info(f"SHAP Force Plot saved at {path}.")
        return path
    except Exception as e:
        logger.error(f"An error occurred while plotting SHAP Force Plot: {e}")
        logger.exception(e)
        raise


def plot_shap_summary(shap_values, X_shap, config, title):
    try:
        plt.figure(figsize=config['plotting']['figure_sizes']['shap_summary'], dpi=config['plotting']['dpi'])
        shap.summary_plot(shap_values, X_shap, show=False)
        plt.title(title, fontsize=14)
        plt.tight_layout()
        path = os.path.join('images', f"{title.replace(' ', '_').lower()}.png")
        plt.savefig(path)
        plt.close()
        logger.info(f"SHAP Summary Plot saved at {path}.")
        return path
    except Exception as e:
        logger.error(f"An error occurred while plotting SHAP Summary: {e}")
        logger.exception(e)
        raise
