import pandas as pd
import polars as pl
import logging
from pathlib import Path
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format=f'{Fore.GREEN}%(asctime)s - %(levelname)s - %(message)s{Style.RESET_ALL}')

class MXMDataLoader:
    def __init__(self, merged_songs, top_words, filtered_tracks):
        self.songs = merged_songs
        self.top_words = top_words
        self.filtered_tracks = filtered_tracks

    def get_sorted_tracks_by_keyword(self, keyword, threshold):
        logging.info(f'Filtering tracks by keyword: {keyword} with a minimum count of {threshold}')
        try:
            keyword_index = self.top_words.index(keyword)
        except ValueError:
            logging.error(Fore.RED + f"Keyword '{keyword}' not found in the dataset.")
            return

        filtered_tracks = []
        for idx, (track_id, word_counts) in enumerate(self.filtered_tracks):
            keyword_count = word_counts.get(keyword_index, 0)
            if keyword_count >= threshold:
                row_df = self.songs.filter(pl.col('track_id') == track_id)
                if len(row_df) > 0:
                    _, artist, title, play_count = row_df[0].row(0)
                    filtered_tracks.append((id, track_id, artist, title, play_count, keyword_count))
        logging.info(Fore.CYAN + "Done filtering tracks by keyword.")
        filtered_tracks_df = pl.DataFrame(filtered_tracks, schema=['index_number', 'track_id', 'artist', 'title', 'play_count', 'keyword_count']).sort('play_count', descending=True).head(50)
        return filtered_tracks_df

def load(mxm_dataset_path):
    logging.info('Loading dataset...')
    top_words = []
    filtered_tracks = []
    with open(mxm_dataset_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith('#') or line.strip() == '':
                continue
            elif line.startswith('%'):
                top_words = line[1:].strip().split(',')
            else:
                elements = line.strip().split(',')
                track_id = elements[0]
                word_counts = {int(count.split(':')[0]) - 1: int(count.split(':')[1]) for count in elements[2:]}
                filtered_tracks.append((track_id, word_counts))
    logging.info('Dataset loaded successfully.')
    return top_words, filtered_tracks

if __name__ == '__main__':
    mxm_dataset_path = 'data/mxm_dataset_train.txt'
    merged_songs_path = 'data/songs.csv'

    if not Path(merged_songs_path).exists():
        logging.info('Merging songs...')
        triplet_columns = ['user_id', 'song_id', 'play_count']
        track_columns = ['track_id', 'song_id', 'artist', 'title']

        triplet_df = pl.read_csv('data/train_triplets.txt', separator='\t', new_columns=triplet_columns, use_pyarrow=True)
        unique_tracks_df = pl.read_csv('data/p02_unique_tracks.csv', new_columns=track_columns)
        triplet_df = triplet_df.group_by('song_id').agg(pl.sum('play_count').alias('play_count')).sort('play_count', descending=True)
        merged_songs = triplet_df.join(unique_tracks_df, on='song_id', how='left').select('track_id', 'artist', 'title', 'play_count')
        merged_songs.write_csv(merged_songs_path)
        logging.info('Songs merged and saved.')
    else:
        logging.info('Reading merged songs...')
        merged_songs = pl.read_csv(merged_songs_path, use_pyarrow=True)

    top_words, filtered_tracks = load(mxm_dataset_path)
    mxm_loader = MXMDataLoader(merged_songs, top_words, filtered_tracks)

    # keyword = 'life'
    # logging.info(f'Looking for tracks with the keyword "{keyword}"...')
    # tracks = mxm_loader.get_sorted_tracks_by_keyword(keyword, 6)
    # if tracks is not None:
    #     print(Fore.YELLOW + "Top tracks containing the keyword 'life':")
    #     print(tracks.to_pandas().to_string(index=False))
    # else:
    #     print(Fore.RED + "No tracks found with the specified keyword and threshold.")




    # loader = MXMDataLoaderClassification(merged_songs, top_words, filtered_tracks)
    # loader.load()
    # categories = {
    #     "love": ["love", "heart"],
    #     "war": ["war", "battle"],
    #     "money": ["money", "cash"],
    #     "loneliness": ["lonely", "alone"]
    # }
    # labeled_tracks = loader.label_tracks(categories)
    # print(labeled_tracks)



# class MXMDataLoaderClassification:
#     def __init__(self, dataset_path_mxm, merged_songs, wv):
#         self.dataset_path = dataset_path_mxm
#         self.songs = merged_songs
#         self.top_words = []
#         self.filtered_tracks = []
#         self.word_vectors = wv

#     def load(self):
#         with open(self.dataset_path, 'r', encoding='utf-8') as file:
#             for line in file:
#                 if line.startswith('#') or line.strip() == '':
#                     continue
#                 elif line.startswith('%'):
#                     self.top_words = line[1:].strip().split(',')
#                 else:
#                     elements = line.strip().split(',')
#                     track_id = elements[0]
#                     word_counts = {int(count.split(':')[0]) - 1: int(count.split(':')[1]) for count in elements[2:]}
#                     self.filtered_tracks.append((track_id, word_counts))

#     def classify_tracks_by_keywords(self, keywords):
#         tracks_classification = []
#         # Create a dictionary to find indices of keywords in top_words
#         keyword_indices = {keyword: self.top_words.index(keyword) for keyword in keywords if keyword in self.top_words}

#         for track_id, word_counts in self.filtered_tracks:
#             keyword_presence = {keyword: word_counts.get(idx, 0) for keyword, idx in keyword_indices.items()}

#             # Determine the most prevalent keyword based on counts
#             if keyword_presence:
#                 most_prevalent_keyword = max(keyword_presence, key=keyword_presence.get)
#                 max_count = keyword_presence[most_prevalent_keyword]
#                 if max_count > 0:  # Only consider tracks where the keyword count is greater than zero
#                     row_df = self.songs.filter(pl.col('track_id') == track_id)
#                     if len(row_df) > 0:
#                         _, artist, title, play_count = row_df[0].row(0)
#                         tracks_classification.append((track_id, artist, title, play_count, most_prevalent_keyword))

#         # Convert to DataFrame
#         classification_df = pl.DataFrame(tracks_classification, schema=['track_id', 'artist', 'title', 'play_count', 'label'])
#         self.classification_df = classification_df
#         return classification_df



# class MXMDataLoaderClassification:
#     def __init__(self, dataset_path_mxm):
#         self.dataset_path = dataset_path_mxm
#         self.top_words = []
#         self.filtered_tracks = []

#     def load(self):
#         with open(self.dataset_path, 'r', encoding='utf-8') as file:
#             for line in file:
#                 if line.startswith('#') or line.strip() == '':
#                     continue
#                 elif line.startswith('%'):
#                     self.top_words = line[1:].strip().split(',')
#                 else:
#                     elements = line.strip().split(',')
#                     track_id = elements[0]
#                     word_counts = {int(count.split(':')[0]) - 1: int(count.split(':')[1]) for count in elements[2:]}
#                     self.filtered_tracks.append((track_id, word_counts))

#     def train(self):
#         X = []
#         y = []

#         for track_id, word_counts in self.filtered_tracks:
#             word_counts_vec = [word_counts.get(idx, 0) for idx in range(len(loader.top_words))]
#             X.append(word_counts_vec[:728])
#             label = self.labeled_tracks.filter(pl.col('track_id') == track_id)['label'].item(0)
#             y.append(label)

#         tfidf_transformer = TfidfTransformer()
#         tfidf_matrix = tfidf_transformer.fit_transform(X)

#         X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, y, test_size=0.2, random_state=42)
#         print("Training and testing data shapes:",  X_train.shape, X_test.shape, len(y_train), len(y_test))

#         classifier = RandomForestClassifier(n_estimators=100, random_state=42)
#         classifier.fit(X_train, y_train)

#         y_pred = classifier.predict(X_test)


#         print(classification_report(y_test, y_pred))

#         return classifier

#     def label_tracks(self, categories):
#         track_labels = []

#         # Map keywords to their indices for quick lookup
#         keyword_to_index = {word: idx for idx, word in enumerate(self.top_words)}

#         for track_id, word_counts in self.filtered_tracks:
#             category_counts = {category: 0 for category in categories}

#             # Accumulate counts for each category based on associated keywords
#             for category, keywords in categories.items():
#                 for keyword in keywords:
#                     idx = keyword_to_index.get(keyword)
#                     if idx is not None:
#                         category_counts[category] += word_counts.get(idx, 0)

#             # Determine the category with the highest count
#             if category_counts:
#                 dominant_category = max(category_counts, key=category_counts.get)
#                 track_labels.append((track_id, dominant_category))

#         labeled_df = pl.DataFrame(track_labels, schema=['track_id', 'label'])
#         self.labeled_tracks = labeled_df
#         return labeled_df


  # ### 3. Classification Task

  # keywords = ["love", "war", "happiness", "loneliness", "money"]
  # loader = MXMDataLoaderClassification(mxm_dataset_path, mergerd_songs)
  # loader.load()

  # classified_tracks = loader.classify_tracks_by_keywords(keywords)
  # classified_tracks

  # classified_tracks['label'].value_counts()
