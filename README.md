# Diabetes Prediction with AI

This project demonstrates a machine learning solution for predicting diabetes based on user-provided health data. The application uses **Streamlit** for an interactive web interface and advanced interpretability tools like SHAP and permutation importance to explain model predictions.

## Live Demo

Check out the live application: [Diabetes Prediction App](https://diabetes-prediction-uz.streamlit.app/)

---

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Features](#features)
4. [Installation](#installation)
5. [How It Works](#how-it-works)
6. [Explanation Methods](#explanation-methods)
7. [Model Performance](#model-performance)
8. [Project Motivation](#project-motivation)
9. [Contributing](#contributing)
10. [License](#license)

---

## Overview

The **Diabetes Prediction with AI** project leverages a machine learning model to predict diabetes risk. Built with **Streamlit**, the app explains predictions using SHAP and permutation importance while showcasing model performance metrics.

### Why This Project?

Understanding diabetes risk through data-driven predictions can help identify potential cases early. This project also demonstrates:
- Practical application of machine learning.
- Model interpretability through SHAP and permutation importance.
- Real-world deployment of machine learning models.

---

## Dataset

The dataset is sourced from the **National Institute of Diabetes and Digestive and Kidney Diseases**. It includes:
- **Pregnancies**
- **Glucose Levels**
- **Insulin Levels**
- **BMI (Body Mass Index)**
- **Age**
- **Outcome**: Indicates diabetes presence (1 = Diabetes, 0 = No Diabetes).

---

## Features

1. **Interactive Input**: Enter health parameters (Pregnancies, Glucose, Insulin, BMI, Age).
2. **Diabetes Prediction**: Real-time risk prediction with probability.
3. **SHAP Explanations**: Visualize individual prediction explanations using:
   - Waterfall Plot
   - Force Plot
4. **Permutation Importance**: Analyze which features most influence the predictions.
5. **Performance Metrics**:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - ROC AUC
6. **Informational Section**: Learn about diabetes risk factors in the "About" section.

---

## Installation

### Prerequisites
- Python 3.8 or above
- Pip package manager

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/UznetDev/Diabetes-Prediction.git
   cd Diabetes-Prediction
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application locally:
   ```bash
   streamlit run main.py
   ```

---

## How It Works

### Application Workflow
1. **User Input**:
   - Enter health data in the sidebar.
   - Features: Pregnancies, Glucose, Insulin, BMI, Age.
2. **Prediction**:
   - The trained model predicts diabetes risk and displays the result.
3. **Explanation**:
   - View SHAP plots (Waterfall and Force) for detailed feature contributions.
   - Explore permutation importance for global feature analysis.
4. **Model Performance**:
   - Metrics such as Accuracy, F1 Score, and ROC AUC are displayed.

### Key Files and Modules
- **`main.py`**: Orchestrates the Streamlit app.
- **`training.py`**: Trains the model and saves it.
- **`loader.py`**: Loads dataset, model, and performance metrics.
- **`input.py`**: Collects user inputs via the sidebar.
- **`predict.py`**: Handles predictions and visualization.
- **`explainer.py`**: Displays SHAP-based explanations.
- **`perm_importance.py`**: Visualizes permutation importance.
- **`performance.py`**: Renders performance metrics.
- **`about.py`**: Displays information about diabetes.
- **`diabetes.csv`**: Dataset file.

---

## Explanation Methods

1. **SHAP Waterfall Plot**:
   - Shows how each feature contributes positively or negatively to the prediction.
2. **SHAP Force Plot**:
   - Interactive visualization of feature contributions to individual predictions.
3. **Permutation Importance**:
   - Ranks features by their impact on the model's predictions.

---

## Model Performance

Performance metrics calculated:
- **Accuracy**: Percentage of correct predictions.
- **Precision**: Ratio of true positives to total positive predictions.
- **Recall**: Ratio of true positives to total actual positives.
- **F1 Score**: Harmonic mean of Precision and Recall.
- **ROC AUC**: Area under the ROC curve.

Metrics are displayed as donut charts in the application.

---

## Project Motivation

This project was developed to:
- Build knowledge in machine learning, especially in healthcare.
- Gain hands-on experience with model interpretability techniques like SHAP.
- Deploy an AI solution using **Streamlit**.

---

## Contributing

Contributions are welcome! Follow these steps:
1. Fork the repository.
2. Create a new feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and push:
   ```bash
   git commit -m "Feature description"
   git push origin feature-name
   ```
4. Submit a pull request.

---

## License

This project is licensed under the MIT License.