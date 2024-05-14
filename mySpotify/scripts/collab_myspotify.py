import os
import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix
import implicit
from implicit.evaluation import precision_at_k
from implicit.nearest_neighbours import CosineRecommender
import logging
from colorama import Fore, Style, init
import warnings
warnings.filterwarnings('ignore')

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format=f'{Fore.GREEN}%(asctime)s - %(levelname)s - %(message)s{Style.RESET_ALL}')

# Initialize colorama
init(autoreset=True)


class MusicData:
    def __init__(self, data):
        self.data = data
        self.song_id_to_name = pd.Series(data.song.values, index=data.song_id).to_dict()

    def get_user_song_names(self, user_id):
        user_data = self.data[self.data['user_id'] == user_id]
        user_songs = [self.song_id_to_name[song_id] for song_id in user_data['song_id'].unique()]
        return user_songs

    def get_user_song_ids(self, user_id):
      user_data = self.data[self.data['user_id'] == user_id]
      return user_data['song_id'].unique()

    def get_song_users(self, song_id):
        song_data = self.data[self.data['song_id'] == song_id]
        song_users = song_data['user_id'].unique()
        return song_users

    def get_songs(self, song_ids):
        return self.data.filter(pl.col('user_idx') == song_ids ).select(['track_id', 'artist', 'title'])

class MusicRecommender:
    def __init__(self, music_data, train_data, test_data):
        self.music_data = music_data
        self.train_data = train_data
        self.test_data = test_data
        self.model : implicit.als.AlternatingLeastSquares = None
        self.model_path = './models/model.npz'
        logging.info('Converting data to COO format...')
        self.train_coo = self._convert_to_coo(self.train_data)
        self.test_coo = self._convert_to_coo(self.test_data)
        logging.info('Data converted successfully.')

    def _convert_to_coo(self, data):
        return coo_matrix((data.play_count.astype(float), (data.index.get_level_values(0), data.index.get_level_values(1),)))

    def _load_model(self):
        logging.info(f'Loading model from {self.model_path}...')
        self.model = self.model.load(self.model_path)
        logging.info('Model loaded successfully.')
        return self.model

    def train(self, factors=728, iterations=50, regularization=0.01, show_progress=True):
      logging.info(f'Model: AlternatingLeastSquares, Factors: {factors}, Iterations: {iterations}, Regularization: {regularization}')
      self.model = implicit.als.AlternatingLeastSquares(factors=factors,
                                                        iterations=iterations,
                                                        regularization=regularization,
                                                        random_state=42,
                                                        )
      logging.info('Checking if model already exists...')
      if os.path.exists(self.model_path):
        return self._load_model()

      logging.info('Training model...')
      self.model.fit(self.train_coo, show_progress=show_progress)
      logging.info('Model trained successfully.')
      return self.model

    def calculate_precision_at_k(self, k=10, num_threads=4):
        p = precision_at_k(self.model, self.train_coo.tocsr(), self.test_coo.tocsr(), K=k, num_threads=num_threads)
        return p

    def recommend_songs_by_user_id(self, user_id, n=10):

        user_idx = music_data.data[music_data.data['user_id'] == user_id].user_idx.values[0]
        print(f">>> Songs recommended for user with ID {user_id}: ")

        songs_ids, scores = self.model.recommend(user_idx, self.test_coo.tocsr()[n], N=n)

        user_songs_ids = music_data.get_user_song_ids(user_id)

        recommended_songs = [song for song in songs_ids if song not in user_songs_ids][:10]
        df = pd.DataFrame(songs[songs['song_idx'].isin(recommended_songs)].drop_duplicates(subset=['song_idx']).song)
        df.index.name = 'index number'

        return df


    def recommend_songs_by_song_id(self, song_id, n=10):
        filtred_songs = songs[songs['song_id'] == song_id]
        song_name, song_idx = filtred_songs.song.iloc[0], filtred_songs.song_idx.iloc[0]
        print(f'Recommendation for {song_name} with ID {song_id}: ')
        itemids, scores = self.model.similar_items(itemid=song_idx)
        df = pd.DataFrame(songs[songs['song_idx'].isin(itemids)].drop_duplicates(subset=['song_idx']).song)
        df.index.name = 'index number'

        return df


def load_data(triplet_path, unique_tracks_path):
    logging.info('Loading data...')


    triplet_columns = ['user_id', 'song_id', 'play_count']
    track_columns = ['track_id', 'song_id', 'artist', 'title']

    triplet_df = pl.read_csv(triplet_path, separator='\t', new_columns=triplet_columns, use_pyarrow=True)
    unique_tracks_df = pl.read_csv(unique_tracks_path, new_columns=track_columns, use_pyarrow=True)

    logging.info('Data loaded successfully.')

    logging.info('Merging songs...')

    triplet_df = triplet_df.filter(pl.col('play_count') > 1)
    songs = pd.merge(triplet_df.to_pandas(), unique_tracks_df.to_pandas(), on='song_id', how='left')
    songs['song'] = songs['title']+' - ' + songs['artist']
    songs = songs[['user_id', 'song_id', 'track_id', 'song', 'play_count']]
    songs['user_idx'] = pd.factorize(songs['user_id'])[0]
    songs['song_idx'] = pd.factorize(songs['song_id'])[0]

    logging.info('Songs merged successfully.')

    del triplet_df, unique_tracks_df

    # save the data
    # songs.to_csv('data/songs.csv', index=False)

    return songs


if __name__ == '__main__':

  # Load data
  global_path = './data'
  triplet_path = f"{global_path}/train_triplets.txt"
  unique_tracks_path = f"{global_path}/p02_unique_tracks.csv"

  songs = load_data(triplet_path, unique_tracks_path)

  music_data = MusicData(songs)


  # Splitting the data into training and testing sets
  X = songs[['user_idx', 'song_idx', 'play_count']]
  train_data, test_data = train_test_split(X, test_size=0.2, random_state=42)
  train_data.set_index(["user_idx", "song_idx"], inplace=True)
  test_data.set_index(["user_idx", "song_idx"], inplace=True)

  # Create MusicRecommender instance
  music_recommender = MusicRecommender(music_data, train_data, test_data)

  # Train the model
  music_recommender.train()

  # Calculate precision at k
  # p = music_recommender.calculate_precision_at_k(k=10, num_threads=4)
  # print(f'>>> >>> Precision at k=10: {p}')

  # Recommend songs based on user ID
  logging.info('People similar to you listen 🎧')
  userId = 'b7815dbb206eb2831ce0fe040d0aa537e2e800f7'
  recommended_songs = music_recommender.recommend_songs_by_user_id(userId, n=10)
  print(recommended_songs)


  # Recommend songs based on song ID
  logging.info('People who listen to this track usually listen👂 🎧')
  songId = 'SOWYSKH12AF72A303A'
  recommended_songs = music_recommender.recommend_songs_by_song_id(songId, n=10)
  print(recommended_songs)
