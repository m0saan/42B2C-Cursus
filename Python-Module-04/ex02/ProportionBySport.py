import pandas as pd
from FileLoader import FileLoader

import pandas as pd

def proportion_by_sport(df: pd.DataFrame, year: int, sport: str, gender: str):
    """
    Returns the proportion of participants who played a given sport among the participants of a given gender.
    
    Args:
    - df: pandas.DataFrame containing the athlete_events.csv dataset.
    - year: an integer representing the Olympic year to consider.
    - sport: a string representing the sport to consider.
    - gender: a string representing the gender to consider ('M' for male, 'F' for female).
    
    Returns:
    - A float representing the proportion (percentage) of participants who played the given sport among the participants of the given gender.
    """
    
    # Select rows corresponding to the given year and sport
    subset = df[(df['Year'] == year) & (df['Sport'] == sport)]
    # Count the number of participants of the given gender
    gender_count = len(subset[subset['Sex'] == gender]['Sex'].unique())
    # Count the total number of participants
    total_count = len(df['Sex'].unique())
    # Calculate the proportion as a percentage
    proportion = total_count / gender_count * 100
    return proportion
    
    

if __name__ == '__main__':
    loader = FileLoader()
    data = loader.load('../data/athlete_events.csv')
    print(proportion_by_sport(data, 2004, 'Tennis', 'F'))
    