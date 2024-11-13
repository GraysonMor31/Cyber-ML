# Cyber-ML

## Project Overview

Cyber-ML is a sophisticated machine learning project designed to detect malware using a hybrid approach that combines Random Forest and Neural Network models. The project leverages the strengths of both models to achieve high accuracy in malware detection.

## Features

- **Data Loading and Preprocessing**: Efficiently load and preprocess large datasets of malware and benign files.
- **Feature Extraction**: Extract meaningful features from raw data to improve model performance.
- **Random Forest Pre-classifier**: Use a Random Forest model to generate initial predictions and enhance feature sets.
- **Neural Network Classifier**: Train a Neural Network model using the enriched feature set for final classification.
- **Model Training and Evaluation**: Comprehensive training and evaluation pipeline to assess model performance.

## Technical Implementation

### Data Loading

- Implemented a `DataLoader` class to handle the loading of malware and benign data from CSV files.
- Utilized `pandas` for efficient data manipulation and handling.

### Feature Extraction and Preprocessing

- Developed a `feature_extractor.py` module to preprocess data, including handling missing values, scaling numerical features, and encoding categorical variables.
- Used `scikit-learn`'s `ColumnTransformer` and `Pipeline` for streamlined preprocessing.

### Random Forest Pre-classifier

- Created a custom `RandomForestPreclassifier` class using `scikit-learn`'s `RandomForestClassifier`.
- Trained the Random Forest model to generate initial predictions, which are then used as additional features for the Neural Network.

### Neural Network Classifier

- Designed a `CyberMLModel` class using `PyTorch` to define the architecture of the Neural Network.
- Implemented training and evaluation functions to optimize the model using backpropagation and assess its performance.

### Model Training and Evaluation

- Combined the original features with the Random Forest predictions to create an enriched feature set.
- Trained the Neural Network using the enriched feature set and evaluated its performance on a test set.

## Setup Instructions

### Prerequisites

- Python 3.6 to 3.9
- pip (Python package installer)

### Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/GraysonMor31/Cyber-ML.git
   cd Cyber-ML
