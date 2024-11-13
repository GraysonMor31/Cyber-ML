import pandas as pd

class DataLoader:
    data_path = 'Data/'
    
    def __init__(self, data_path):
        self.data_path = data_path
    
    def load_malware_data(self):
        try:
            malware_df = pd.read_csv(self.data_path + 'malware.csv')
            print(malware_df.head())
            return malware_df
        except FileNotFoundError:
            print(f"Error: File '{self.data_path + 'malware.csv'}' not found.")
        except pd.errors.EmptyDataError:
            print(f"Error: File '{self.data_path + 'malware.csv'}' is empty.")
    
    def load_benign_data(self):
        try:
            benign_df = pd.read_csv(self.data_path + 'benign.csv')
            print(benign_df.head())
            return benign_df
        except FileNotFoundError:
            print(f"Error: File '{self.data_path + 'benign.csv'}' not found.")
        except pd.errors.EmptyDataError:
            print(f"Error: File '{self.data_path + 'benign.csv'}' is empty.")