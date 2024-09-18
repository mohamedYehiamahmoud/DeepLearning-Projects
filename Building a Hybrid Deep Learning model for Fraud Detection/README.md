
# Credit Card Application Analysis

This script analyzes credit card applications using machine learning techniques, including both unsupervised and supervised learning approaches.

## Overview

The script performs the following main tasks:
1. Loads and preprocesses credit card application data
2. Applies Self-Organizing Map (SOM) for unsupervised fraud detection
3. Trains an Artificial Neural Network (ANN) for supervised classification

## Dependencies

- numpy
- matplotlib
- pandas
- sklearn
- minisom
- keras

## Data

The script uses the 'Credit_Card_Applications.csv' file, which contains various features of credit card applications and their approval status.

## Unsupervised Learning (SOM)

1. Data is loaded and scaled using MinMaxScaler
2. A Self-Organizing Map is trained on the scaled data
3. The SOM is visualized to identify potential fraudulent applications
4. Fraudulent applications are extracted based on SOM results

## Supervised Learning (ANN)

1. A new feature matrix is created, including a fraud indicator
2. Data is scaled using StandardScaler
3. An Artificial Neural Network is constructed and trained
4. The ANN predicts fraud probabilities for all applications

## Output

The script generates:
1. A visualization of the SOM results
2. Predictions of fraud probabilities for all applications

## Usage

Ensure all dependencies are installed, then run the script:

```
python my_mega_case_study.py
```

Note: Adjust hyperparameters and model architectures as needed for optimal performance.
"""
