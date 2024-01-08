# Overcoming Long Inference Time of Nearest Neighbours in Regression and Uncertainty Prediction

## Introduction
This repository contains the implementation of the experiments and methods described in our paper "Overcoming Long Inference Time of Nearest Neighbours Analysis in Regression and Uncertainty Prediction" by Alexander Kovalenko, František Koutenský, Petr Šimánek, and Miroslav Čepek, published by the Faculty of Information Technology, Czech Technical University in Prague.

Our work addresses the challenge of high inference time complexity in nearest neighbor analysis. We propose an innovative approach combining k-nearest neighbors (k-NN) and gradient-boosted regression trees, significantly enhancing inference speed while maintaining accuracy. This implementation is inspired by the work of Lowd and Brophy (available at [this GitHub repository](https://github.com/jjbrophy47/ibug)), and we have utilized datasets and metrics similar to those used in their research. We extend our gratitude to them for their foundational work, which has significantly influenced our approach.

## Module Descriptions

### `data_handler.py`
- **Purpose**: Handles preprocessed data loading and result management for experiments.

### `metrics.py`
- **Description**: Provides functions for calculating and visualizing various metrics related to location (point predictions) and scale (uncertainty estimates) in regression tasks.

### `parameters.py`
- **Description**: Defines global parameters and constants used throughout the project.

### `location_methods.py`
- **Description**: Contains the abstract base class `LocationMethod` and its implementations for various location prediction methods experiments, with CatBoost as a specific example.

### `uncertainty_methods.py`
- **Description**: Implements various methods for uncertainty estimation in regression models, providing an abstract base class and multiple concrete implementations.
- **Classes and Key Methods**:
  - `UncertaintyMethod` (Abstract Base Class): Defines the structure for uncertainty estimation methods.
  - `CBU` (inherits `UncertaintyMethod`): Implements uncertainty estimation using CatBoost with uncertainty (`RMSEWithUncertainty` loss function).
  - `IBUG` (inherits `UncertaintyMethod`): Integrates the IBUG method for uncertainty estimation.
  - `kNN` (inherits `UncertaintyMethod`): Applies a k-Nearest Neighbors approach for uncertainty estimation (using [FAISS library](https://github.com/facebookresearch/faiss)).
  - `CB` (inherits `UncertaintyMethod`): Uses a CatBoost Regressor for uncertainty estimation. In this scenario, the labels for the CatBoost were created as the residuals after location predictions (the difference between true and predicted price, in case of car appraisal). 
  - `AcceleratedUncertaintyMethod` (inherits `UncertaintyMethod`): This class represents the developed method from the paper (CatBoost + kNN). It takes trained uncertainty predictor, creates the uncertainty labels for the training data with that predictor, and uses CatBoost to be trained on these labels.
    
### `experiments.py`
- **Description**: Orchestrates the execution of location and uncertainty experiments, encompassing the entire workflow from data handling to result generation.
- **Functions**:
  - `get_uncertainty_methods`: Initializes and returns a list of uncertainty estimation methods with their corresponding parameters.
  - `run_loc_experiment`: Executes the location prediction experiment. It tunes, trains, and evaluates location prediction methods on various datasets, saving the results and models.
  - `run_unc_experiment`: Carries out the uncertainty estimation experiment. This involves loading location predictors, tuning, training, and evaluating uncertainty methods on test data, and saving the results.
- **Output**:
  - The script generates CSV files with aggregated experiment results, including metrics for location and scale predictions, training and prediction times, and plots for uncertainty evaluation.
    
### `data/`
- **Description**: This directory includes folders for each dataset used in the experiments. Each folder contains instructions on how to download and preprocess the respective dataset into a format compatible with the experiment code. The preprocessing scripts and methodologies are derived from the work by Brophy and Lowd in their ibug project.
- **Contents**:
  - Individual folders for each dataset.
  - Step-by-step guides and scripts for data acquisition and preprocessing.
  - Note: The data preprocessing scripts and procedures in this section are directly taken from the IBUG project by Brophy and Lowd.


Thank you for your interest in our research and implementation. For any further questions or collaboration, feel free to reach out to the authors.
