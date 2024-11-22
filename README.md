Here's an enhanced and more comprehensive version of the README file with details about the dataset, parallel processing techniques, and GPU utilization:

---

# Heart Disease Prediction with Explainability

This project implements a comprehensive pipeline for binary classification to predict heart disease. The pipeline leverages both machine learning and deep learning techniques and integrates modern tools for data preprocessing, exploratory analysis, hyperparameter optimization, model explainability, and GPU-based acceleration. It also emphasizes reproducibility and scalability.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Modules Overview](#modules-overview)
- [Results](#results)
- [Contributing](#contributing)

---

## Overview

The project focuses on heart disease prediction using binary classification (`1` for heart disease, `0` for no heart disease). It employs advanced techniques for:
- Handling class imbalance.
- Hyperparameter tuning using Keras Tuner (Hyperband).
- Model explainability with SHAP (SHapley Additive exPlanations).
- Parallel and GPU-based computations for efficient training.

---

## Features

- **Data Preprocessing**: Missing value handling, feature scaling, and train-test splitting.
- **Exploratory Data Analysis**: Visualization tools for understanding data distributions and correlations.
- **Deep Learning Model**: Customizable neural network with tunable architecture and learning parameters.
- **Hyperparameter Optimization**: Keras Tuner for identifying the optimal model configuration.
- **Explainability**: SHAP integration for understanding feature importance and individual predictions.
- **Parallel Processing and GPU Utilization**: Efficient data processing and accelerated model training using TensorFlow's GPU support.
- **Comprehensive Reporting**: Automatically generated PDF reports summarizing metrics, plots, and insights.

---

## Dataset

The project uses the **Heart Disease Dataset** (commonly referred to as the Cleveland dataset), which contains medical data for heart disease diagnosis. Here's a summary of the dataset:

- **Source**: UCI Machine Learning Repository
- **Features**:
  - Demographic data (e.g., age, sex)
  - Clinical data (e.g., cholesterol levels, blood pressure)
  - Results from medical tests (e.g., ECG, thalach)
  - Lifestyle indicators (e.g., exercise-induced angina)
- **Target Variable**: Binary label (`0`: No heart disease, `1`: Heart disease).
- **Size**: 303 samples with 14 features.

You can customize the dataset path in the `config.yaml` file.

---

## Technologies Used

### **Parallel Processing and GPU Utilization**
1. **TensorFlow GPU Support**:
   - The project is optimized for GPU usage, enabling accelerated model training.
   - TensorFlow's `tf.config.experimental.set_memory_growth` ensures efficient memory allocation.
   - Multi-threaded data loading and preprocessing are leveraged for speedup.

2. **Parallel Processing for Tuning**:
   - Hyperparameter tuning uses Keras Tuner's `Hyperband`, which parallelizes multiple trials.
   - TensorFlow seamlessly integrates with hardware acceleration for tuning and evaluation.

3. **Multi-core Processing**:
   - NumPy and Pandas operations utilize multi-threading for fast computations.
   - Scikit-learn functions for data splitting and scaling are optimized for parallel execution.

---

## Project Structure

```plaintext
├── config.yaml                 # Configuration file
├── heart.csv                   # Dataset
├── main.py                     # Entry point of the project
├── modules/
│   ├── __init__.py             # Initialization script
│   ├── class_imbalance.py      # Handles class imbalance
│   ├── data_loader.py          # Loads dataset
│   ├── exploratory.py          # EDA module
│   ├── missing_values.py       # Handles missing values
│   ├── model_builder.py        # Builds and compiles models
│   ├── plotter.py              # Generates plots for metrics
│   ├── preprocessing.py        # Data preprocessing utilities
│   ├── report_generator.py     # Generates PDF reports
│   ├── trainer.py              # Trains the model
│   ├── evaluator.py            # Evaluates the model
│   ├── hyperparameter_tuner.py # Hyperparameter tuning
│   ├── interpretability.py     # SHAP-based model explainability
├── images/                     # Output directory for plots
├── reports/                    # Output directory for reports
└── logs/                       # Logs directory
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda for package management
- A CUDA-enabled GPU and compatible NVIDIA drivers (optional for GPU acceleration)

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/truedashti/heart-disease-prediction.git
   cd heart-disease-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. For GPU support, ensure TensorFlow GPU is installed:
   ```bash
   pip install tensorflow-gpu
   ```

---

## Usage

1. Configure parameters in `config.yaml`.
2. Run the main script:
   ```bash
   python main.py
   ```

The script will:
- Load and preprocess the dataset.
- Train the model with optimized hyperparameters.
- Generate plots and evaluation metrics.
- Create a PDF report summarizing the results.

---

## Modules Overview

### **Data Preprocessing**
- **`data_loader.py`**: Loads the dataset.
- **`missing_values.py`**: Handles missing values with strategies like drop, mean, median, or mode.
- **`preprocessing.py`**: Scales features and splits data into training and testing sets.

### **Exploratory Data Analysis**
- **`exploratory.py`**: Generates EDA plots such as target distribution and correlation heatmaps.

### **Model Training and Evaluation**
- **`model_builder.py`**: Builds customizable deep learning models.
- **`trainer.py`**: Trains the model with callbacks for early stopping and F1 score tracking.
- **`evaluator.py`**: Evaluates models using various metrics.

### **Explainability**
- **`interpretability.py`**: Generates SHAP values for feature importance and prediction explanations.
- **`plotter.py`**: Visualizes SHAP summary, force, and waterfall plots.

### **Hyperparameter Optimization**
- **`hyperparameter_tuner.py`**: Optimizes the model architecture and training parameters.

### **Report Generation**
- **`report_generator.py`**: Compiles all metrics, plots, and SHAP explanations into a comprehensive PDF report.

---

## Results

The project outputs include:
- **Metrics**: Accuracy, precision, recall, F1 score, and ROC-AUC.
- **Plots**: 
  - Confusion matrix
  - ROC curve
  - Precision-recall curve
  - SHAP visualizations
- **Reports**: Auto-generated PDF with a detailed analysis.

---

## Contributing

Contributions are welcome! If you find issues or want to add features, feel free to open issues or submit pull requests on [GitHub](https://github.com/truedashti).

