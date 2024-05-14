import pandas as pd
import polars as pl
import logging
from pathlib import Path
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Define custom log formatter to include color
class ColorLogFormatter(logging.Formatter):
    def __init__(self):
        super().__init__('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    def format(self, record):
        log_colors = {
            'DEBUG': Fore.CYAN,
            'INFO': Fore.GREEN,
            'WARNING': Fore.YELLOW,
            'ERROR': Fore.RED,
            'CRITICAL': Fore.RED + Style.BRIGHT
        }
        log_color = log_colors.get(record.levelname, Fore.WHITE)
        record.msg = log_color + str(record.msg) + Style.RESET_ALL
        return super().format(record)

# Configure logging
logger = logging.getLogger()
handler = logging.StreamHandler()
handler.setFormatter(ColorLogFormatter())
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def load_data(triplet_path, unique_tracks_path, genre_path):
    logging.info('Loading data...')
    triplet_columns = ['user_id', 'song_id', 'play_count']
    track_columns = ['track_id', 'song_id', 'artist', 'title']
    genre_column_names = ['track_id', 'majority_genre', 'minority_genre']

    triplet_df = pl.read_csv(triplet_path, separator='\t', new_columns=triplet_columns, use_pyarrow=True)
    unique_tracks_df = pl.read_csv(unique_tracks_path, new_columns=track_columns, use_pyarrow=True)
    genre_df = pd.read_csv(genre_path, sep='\t', comment='#', names=genre_column_names)

    logging.info('Data loaded successfully.')
    return triplet_df, unique_tracks_df, genre_df.drop(columns=['minority_genre'])

def get_top_tracks(triplet_df, unique_tracks_df, n=250):
    logging.info('Calculating top tracks...')
    song_play_counts = triplet_df.group_by('song_id').agg(pl.sum('play_count').alias('play_count')).sort('play_count', descending=True).limit(n)
    top_tracks = song_play_counts.join(unique_tracks_df, on='song_id').select(['artist', 'title', 'play_count']).with_row_index(name='index number')
    logging.info('Top tracks calculation completed.')
    return top_tracks.sort('play_count', descending=True)

def get_top_genre_tracks(unique_tracks_df, genre_df, song_play_counts, selected_genre, n=100):
    logging.info(f'Processing top tracks for genre: {selected_genre}')
    unique_tracks_df_pandas = unique_tracks_df.to_pandas()
    merged_df = pd.merge(pd.merge(genre_df, unique_tracks_df_pandas, on='track_id'), song_play_counts, on='song_id')

    genre_subset = merged_df[merged_df['majority_genre'] == selected_genre]
    track_play_counts = genre_subset.groupby(['artist', 'title'])['play_count'].sum().sort_values(ascending=False).head(n)
    top_tracks = track_play_counts.head(5)
    bottom_tracks = track_play_counts.tail(5)
    logging.info(f'Top and bottom tracks for genre {selected_genre} processed.')
    return top_tracks, bottom_tracks

def main():
    global_path = './data'
    triplet_path = f"{global_path}/train_triplets.txt"
    unique_tracks_path = f"{global_path}/p02_unique_tracks.csv"
    genre_path = f"{global_path}/p02_msd_tagtraum_cd2.cls"

    triplet_df, unique_tracks_df, genre_df = load_data(triplet_path, unique_tracks_path, genre_path)
    song_play_counts = triplet_df.group_by('song_id').agg(pl.sum('play_count').alias('play_count')).to_pandas()

    top_250_tracks = get_top_tracks(triplet_df, unique_tracks_df)
    logging.info("Top 250 tracks processed.")
    print(Fore.BLUE + "Top 5 tracks:", top_250_tracks.head(5))
    print(Fore.BLUE + "Bottom 5 tracks:", top_250_tracks.tail(5))

    selected_genres = ['Rock', 'Rap', 'Electronic']
    for selected_genre in selected_genres:
        top_tracks, bottom_tracks = get_top_genre_tracks(unique_tracks_df, genre_df, song_play_counts, selected_genre)
        print(Fore.BLUE + f"Top 5 tracks for the genre {selected_genre}:", top_tracks)
        print(Fore.BLUE + f"Bottom 5 tracks for the genre {selected_genre}:", bottom_tracks)
        print(Fore.YELLOW + '----------------------------------------------------------------------\n')

if __name__ == "__main__":
    main()
