import pandas as pd
from data_loader import DataLoader
from random_forest import RandomForestPreclassifier
from sklearn.model_selection import train_test_split

def load_data():
    data_loader = DataLoader(data_path='Data/')
    malware_df = data_loader.load_malware_data()
    benign_df = data_loader.load_benign_data()
    return malware_df, benign_df

def setup_rf_classifier():
    rf_classifier = RandomForestPreclassifier(n_estimators=100, random_state=42)
    return rf_classifier

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
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Setup the Random Forest classifier
    rf_classifier = setup_rf_classifier()
    
    # Train the classifier
    rf_classifier.fit(X_train, y_train)
    
    # Evaluate the classifier
    evaluation_metrics = rf_classifier.evaluate(X_test, y_test)
    print(evaluation_metrics)

if __name__ == '__main__':
    main()