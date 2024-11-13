import pandas as pd

class DataLoader:
    data_path = 'Data/'
    
    def __init__(self, data_path):
        self.data_path = data_path
    
    # Load the data from the first CSV file (malware)
    def load_malware_data(self):
        try:
            malware_df = pd.read_csv(self.data_path + 'malware.csv')
        except FileNotFoundError:
            print(f"Error: File '{self.data_path + 'malware.csv'}' not found.")
        except pd.errors.EmptyDataError:
            print(f"Error: File '{self.data_path + 'malware.csv'}' is empty.")
        print(malware_df.head())
        
    # Load the data from the second CSV file (benign)
    def load_benign_data(self):
        try:
            benign_df = pd.read_csv(self.data_path + 'benign.csv')
        except FileNotFoundError:
            print(f"Error: File '{self.data_path + 'benign.csv'}' not found.")
        except pd.errors.EmptyDataError:
            print(f"Error: File '{self.data_path + 'benign.csv'}' is empty.")
        print(benign_df.head())