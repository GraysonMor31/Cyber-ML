import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Define preprocessing for numerical columns (impute missing values and scale)
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Define preprocessing for categorical columns (impute missing values and one-hot encode)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # Apply preprocessing
    df_preprocessed = preprocessor.fit_transform(df)
    
    return df_preprocessed, preprocessor

def extract_features(df):
    # Example feature extraction logic
    # Add your custom feature extraction logic here
    df['feature1'] = df['column1'] * df['column2']
    df['feature2'] = df['column3'] / (df['column4'] + 1)
    
    return df

def preprocess_and_extract_features(df):
    # Extract features
    df = extract_features(df)
    
    # Preprocess data
    df_preprocessed, preprocessor = preprocess_data(df)
    
    return df_preprocessed, preprocessor