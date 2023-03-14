import pandas as pd
from FileLoader import FileLoader

def how_many_medals_by_country(df: pd.DataFrame, country: str) -> dict:
    """
    This function takes a pandas DataFrame and a country name as input, and returns a dictionary of dictionaries giving the
    number and type of medal for each competition where the country delegation earned medals.
    
    Args:
    - df: a pandas.DataFrame which contains the dataset
    - country: a string representing the name of the country
    
    Returns:
    - A dictionary of dictionaries giving the number and type of medal for each competition where the country delegation
      earned medals. The keys of the main dictionary are the Olympic games' years. In each year's dictionary, the key are 'G',
      'S', 'B' corresponding to the type of medals won.
    """
    
    # Filter the DataFrame to only include rows where the country earned a medal
    filtered_df = df.loc[df['Team'] == country]
    
    # Group the filtered DataFrame by Year and Medal
    grouped_df = filtered_df.groupby(['Year', 'Medal'])
    
    # Create a dictionary to store the results
    results = {}
    
    # Loop through the groups and count the number of medals for each year and medal type
    for group, data in grouped_df:
        year = group[0]
        medal_type = group[1]
        if year not in results:
            results[year] = {'G': 0, 'S': 0, 'B': 0}
        results[year][medal_type] += 1
    
    return results


if __name__ == '__main__':
    loader = FileLoader()
    data = loader.load('../data/athlete_events.csv') # Output
    print(how_many_medals_by_country(data, 'Martian Federation'))