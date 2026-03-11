import torch
import pandas as pd
import numpy as np

class MovieLensData:
    def __init__(self, data_dir):
        # Read rating.csv and sort by userId and timestamp
        df_ratings = pd.read_csv(f"{data_dir}/ratings.csv")
        df_ratings = df_ratings[df_ratings['rating'] >= 3.5]

        df_ratings = df_ratings.sort_values(by=['userId', 'timestamp'])

        # Get uniqueId and apply for user and item
        u_codes, u_uniques = pd.factorize(df_ratings['userId'], sort=False)
        i_codes, i_uniques = pd.factorize(df_ratings['movieId'], sort=False)

        self.num_users = len(u_uniques)
        self.num_items = len(i_uniques)

        # Map item and user
        self.item_map = {raw_id: idx for idx, raw_id in enumerate(i_uniques)}
        self.user_map = {raw: idx for idx, raw in enumerate(u_uniques)}
        self.idx_to_item_raw = {idx: raw_id for idx, raw_id in enumerate(i_uniques)}

        # Add the new idx columns
        df_ratings['u_idx'] = u_codes
        df_ratings['i_idx'] = i_codes

        # Get the last item consumed for each user as Test set
        self.test_df = df_ratings.groupby('u_idx').tail(1)
        self.train_df = df_ratings.drop(self.test_df.index)

        self.interactions = self.train_df[['u_idx', 'i_idx']].values.astype(np.int32)

        grouped = df_ratings.groupby('u_idx')['i_idx'].apply(set)
        self.user_history = grouped.to_dict()

        df_movies = pd.read_csv(f"{data_dir}/movies.csv")
        self.movie_titles = {}

        for _, row in df_movies.iterrows():
            iid = row['movieId']
            if iid in self.item_map:
                internal_idx = self.item_map[iid]
                self.movie_titles[internal_idx] = row['title']

    # Obtain the batch for the training
    def get_batch(self, batch_size, device):
        # Random interactions
        idx = np.random.randint(0, len(self.interactions), size=batch_size)
        batch_pairs = self.interactions[idx]

        # User batch
        u_batch = batch_pairs[:, 0]
        # Positive item batch
        i_batch = batch_pairs[:, 1]
        # Random negative item batch
        j_batch = np.random.randint(0, self.num_items, size=batch_size, dtype=np.int32)

        for k in range(batch_size):
            u = u_batch[k]
            history = self.user_history.get(u, set())
            while j_batch[k] in history:
                j_batch[k] = np.random.randint(0, self.num_items)

        return (
            torch.tensor(u_batch, dtype=torch.long, device=device),
            torch.tensor(i_batch, dtype=torch.long, device=device),
            torch.tensor(j_batch, dtype=torch.long, device=device)
        )