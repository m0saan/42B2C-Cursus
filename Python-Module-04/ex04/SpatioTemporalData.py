import pandas as pd
from FileLoader import FileLoader

class SpatioTemporalData:
    
    def __init__(self, data):
        self.data = data

    def when(self, location):
        years = self.data[self.data['City'] == location]['Year'].unique()
        return list(years)

    def where(self, date):
        location = self.data[self.data['Year'] == date]['City'].unique()
        return location[0] if len(location) > 0 else None
    
if __name__ == '__main__':
    loader = FileLoader()
    data = loader.load('../data/athlete_events.csv') # Output Loading dataset of dimensions 271116 x 15
    sp = SpatioTemporalData(data)
    print(sp.where(1896)) # Output [’Athina’]
    print(sp.where(2016)) # Output [’Rio de Janeiro’]
    print(sp.when('Athina')) # Output [2004, 1906, 1896]
    print(sp.when('Paris')) # Output [1900, 1924]
        