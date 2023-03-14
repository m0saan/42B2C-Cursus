import pandas as pd

class FileLoader:
    def load(self, path):
        try:
            data = pd.read_csv(path)
            print(f"Loading dataset of dimensions {data.shape[0]} x {data.shape[1]}")
            return data
        except:
            print("Error: could not load file")
            return None
        
    def display(self, df, n):
        try:
            if n > 0:
                print(df.head(n))
            else:
                print(df.tail(-n))
        except:
            print("Error: could not display data")