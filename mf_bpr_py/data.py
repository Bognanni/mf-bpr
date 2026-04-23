import torch
import pandas as pd
import numpy as np
import json
from typing import Optional


# parent class for different datasets
class BaseRecSysData:
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

# child class for Amazon All Beauty
class AmazonBeautyData(BaseRecSysData):
    def __init__(self, jsonl_path: str, meta_jsonl_path: Optional[str] = None):
        print(f"Loading dataset from {jsonl_path}.")

        # open the file and read the reviews
        records = []
        with open(jsonl_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                records.append({
                    'userId': data['user_id'],
                    'itemId': data['parent_asin'],  # ASIN
                    'rating': float(data['rating']),
                    'timestamp': int(data['timestamp'])
                })

        df_ratings = pd.DataFrame(records)

        # item considered consumed if rating is >= 3.5
        df_ratings = df_ratings[df_ratings['rating'] >= 3.5]
        df_ratings = df_ratings.sort_values(by=['userId', 'timestamp'])

        # create unique user and item ids
        u_codes, u_uniques = pd.factorize(df_ratings['userId'], sort=False)
        i_codes, i_uniques = pd.factorize(df_ratings['itemId'], sort=False)

        self.num_users = len(u_uniques)
        self.num_items = len(i_uniques)

        # conversion maps
        self.item_map = {raw_id: idx for idx, raw_id in enumerate(i_uniques)}
        self.user_map = {raw: idx for idx, raw in enumerate(u_uniques)}
        self.idx_to_item_raw = {idx: raw_id for idx, raw_id in enumerate(i_uniques)}

        df_ratings['u_idx'] = u_codes
        df_ratings['i_idx'] = i_codes

        # train test split (leave one out)
        self.test_df = df_ratings.groupby('u_idx').tail(1)
        self.train_df = df_ratings.drop(self.test_df.index)

        self.interactions = self.train_df[['u_idx', 'i_idx']].values.astype(np.int32)
        grouped = df_ratings.groupby('u_idx')['i_idx'].apply(set)
        self.user_history = grouped.to_dict()

        # dict to have title of the items
        self.item_titles = {}

        if meta_jsonl_path:
            print(f"Extract the title of the items from: {meta_jsonl_path}.")
            # map asin to title
            asin_to_title = {}

            with open(meta_jsonl_path, 'r') as meta_file:
                for line in meta_file:
                    meta_data = json.loads(line)
                    asin = meta_data.get('parent_asin')

                    # if the asin is in the filtered item map
                    if asin in self.item_map:
                        # get the title if in the meta file
                        title = meta_data.get('title', f"Missing title (ASIN: {asin})")
                        asin_to_title[asin] = title

            # assign the titles
            for idx, raw_id in self.idx_to_item_raw.items():
                self.item_titles[idx] = asin_to_title.get(raw_id, f"Unknown item (ASIN: {raw_id})")

        else:
            # if there is not a meta file
            for idx, raw_id in self.idx_to_item_raw.items():
                self.item_titles[idx] = f"Amazon item ASIN: {raw_id}"

        print(
            f"Extraction done! Users: {self.num_users} | Items: {self.num_items} | Training interactions: {len(self.interactions)}")


# child class for MovieLens
class MovieLensData(BaseRecSysData):
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
        self.item_titles = {}

        for _, row in df_movies.iterrows():
            iid = row['movieId']
            if iid in self.item_map:
                internal_idx = self.item_map[iid]
                self.item_titles[internal_idx] = row['title']
