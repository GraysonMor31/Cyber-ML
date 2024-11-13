import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_loader import DataLoader
from model import initialize_model, train_model, evaluate_model
from random_forest import RandomForestPreclassifier

def load_data():
    data_loader = DataLoader(data_path='Data/')
    malware_df = data_loader.load_malware_data()
    benign_df = data_loader.load_benign_data()
    return malware_df, benign_df

def main():
    # Load the data
    malware_df, benign_df = load_data()
    
    # Combine the data and create labels
    malware_df['label'] = 1
    benign_df['label'] = 0
    data_df = pd.concat([malware_df, benign_df])
    
    # Prepare features and labels
    X = data_df.drop(columns=['label', 'hash'])
    y = data_df['label']
    
    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the Random Forest model
    rf_model = RandomForestPreclassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Get predictions from the Random Forest model
    rf_train_preds = rf_model.predict(X_train).reshape(-1, 1)
    rf_test_preds = rf_model.predict(X_test).reshape(-1, 1)
    
    # Combine the original features with the Random Forest predictions
    X_train_combined = torch.tensor(pd.concat([pd.DataFrame(X_train), pd.DataFrame(rf_train_preds)], axis=1).values, dtype=torch.float32)
    X_test_combined = torch.tensor(pd.concat([pd.DataFrame(X_test), pd.DataFrame(rf_test_preds)], axis=1).values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    
    # Initialize the neural network model, loss function, and optimizer
    input_dim = X_train_combined.shape[1]
    hidden_dim = 64
    output_dim = 1
    learning_rate = 0.001
    nn_model, criterion, optimizer = initialize_model(input_dim, hidden_dim, output_dim, learning_rate)
    
    # Train the neural network model
    train_model(nn_model, criterion, optimizer, X_train_combined, y_train)
    
    # Evaluate the neural network model
    evaluate_model(nn_model, X_test_combined, y_test)

if __name__ == '__main__':
    main()