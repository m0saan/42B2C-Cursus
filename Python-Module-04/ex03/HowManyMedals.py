import pandas as pd
from FileLoader import FileLoader

def how_many_medals(df: pd.DataFrame, participant: str): # -> Dict[str, Dict[str, int]]:
    """
    Returns a dictionary of dictionaries giving the number and type of medals for each year
    during which the participant won medals.

    Args:
    - df: A pandas.DataFrame containing the dataset.
    - participant: A string with the name of the participant.

    Returns:
    A dictionary of dictionaries where the keys of the main dictionary are the Olympic games years,
    and in each year’s dictionary, the keys are 'G', 'S', 'B' corresponding to the type of medals
    won (gold, silver, bronze). The innermost values correspond to the number of medals of a given
    type won for a given year.
    """
    
    medals = {}
    athlete_df = df[df['Name'] == participant].dropna().reset_index()
    years = athlete_df['Year'].unique()
    
    for year in years:
        year_df = athlete_df[athlete_df['Year'] == year]
        medal_counts = {'G': 0, 'S': 0, 'B': 0}

        for medal in ['Gold', 'Silver', 'Bronze']:
            count = len(year_df[year_df['Medal'] == medal])
            medal_counts[medal[0]] = count

        medals[year] = medal_counts

    return medals
    
    
if __name__ == '__main__':
    loader = FileLoader()
    data = loader.load('../data/athlete_events.csv') # Output
    print(how_many_medals(data, 'Kjetil Andr Aamodt'))