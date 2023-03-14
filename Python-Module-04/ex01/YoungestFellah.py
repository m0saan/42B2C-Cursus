import pandas as pd
from typing import Dict, Union
from FileLoader import FileLoader

def youngest_fellah(df: pd.DataFrame, year: int) -> Dict[str, Union[int, None]]:
    """
    Return a dictionary containing the age of the youngest woman and man who took part in the Olympics in the given year.

    Args:
    - data (pd.DataFrame): A pandas dataframe containing the Olympic athletes data.
    - year (int): The year of the Olympics to consider.

    Returns:
    - A dictionary with keys "youngest_woman" and "youngest_man" whose values are the age of the youngest woman and
      man who took part in the Olympics in the given year. If no woman or man participated in the Olympics in the given
      year, the corresponding value will be None.
    """
    
    year_df = df[df['Year'] == year]
    men_df = year_df[year_df['Sex'] == 'M']
    women_df = year_df[year_df['Sex'] == 'F']
    
    youngest_man_age = men_df['Age'].min()
    youngest_woman_age = women_df['Age'].min()
    
    result = {'youngest_man_age': youngest_man_age, 'youngest_woman_age': youngest_woman_age}
    return result


if __name__ == '__main__':
    loader = FileLoader()
    data = loader.load('../data/athlete_events.csv') # Output
    print(youngest_fellah(data, 2004))
    # Output {’f’: 13.0, ’m’: 14.0}