# Neuro 140

This repository contains code and data for predicting future ad performance using early-stage data. The project uses a combination of manual features, image embeddings, and various machine learning models to make predictions.

## Project Structure

### Main Notebook
The primary work is contained in `Modeling.ipynb`, which is organized into the following sections:

#### Data Exploration and Preparation
- **Data filtering**: Initial data cleaning and filtering operations
- **Load dataset**: Code for loading and preparing the dataset
- **Dataset stats**: Statistical analysis and exploratory visualizations of the dataset

#### Initial Experiments (kept for reference, but didn't work)
- **Dataset Loader**: Initial data loading implementation
- **Model definitions**: Early model architectures
- **Experiments**: First round of experiments (pre-cleaning and manual features)

#### Primary Results
The main results are in the "Use early days to predict future performance" section, which includes:

##### Setup
- **Data set up**: Preparation of training and test datasets
- **Model training**: Common training configurations and utilities

##### Models and Experiments
- **Baseline**: Simple baseline using training set average metrics
- **LightGBM Experiments**:
  - Manual features
  - Manual features + PCA image features
  - Manual features + Raw embeddings
  - Manual features + Embedding summaries
  - Manual features + L1 feature selection (unfinished)
- **Two-tower Models**:
  - MLP+MLP architecture
  - MLP+Transformer architecture

### Supporting Directories

#### `scripts/`
Contains utility scripts for:
- Image parsing and processing
- Dataset filtering
- Image embedding generation (including DeepSeek embeddings)

#### `data/`
Contains all data files:
- Full training and test CSV files
- Filtered training and test CSV files
- Image embedding files (to be loaded into the notebook)
