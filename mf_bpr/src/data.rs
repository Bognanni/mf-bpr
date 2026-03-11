use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use rand::Rng;
use candle_core::{Device, Tensor};

// Struct to handle the dataset preprocessed
pub struct MovieLensData {
    pub num_users: usize,
    pub num_items: usize,
    pub interactions: Vec<(usize, usize)>,
    pub user_history: Vec<HashSet<usize>>,
    pub test_set: Vec<(usize, usize)>,
    pub movie_titles: HashMap<usize, String>,
    pub item_raw_to_idx: HashMap<u32, usize>,
    pub user_raw_to_idx: HashMap<u32, usize>,
    pub idx_to_item_raw: HashMap<usize, u32>,
}

struct RawRating {
    uid: u32,
    iid: u32,
    timestamp: u64,
}

impl MovieLensData {
    pub fn load<P: AsRef<Path>>(dir_path: P) -> anyhow::Result<Self> {
        let dir = dir_path.as_ref();
        
        let ratings_path = dir.join("ratings.csv");
        let file = File::open(ratings_path)?;
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(BufReader::new(file));

        let mut raw_data: Vec<RawRating> = Vec::new();

        // Save the raw data using RawRating if the rating is >= 3.5
        for result in rdr.records() {
            let record = result?;
            let rating: f32 = record[2].parse()?;
            
            if rating >= 3.5 {
                raw_data.push(RawRating {
                    uid: record[0].parse()?,
                    iid: record[1].parse()?,
                    timestamp: record[3].parse()?,
                });
            }
        }

        // Sort by user id and timestamp
        raw_data.sort_by_key(|r| (r.uid, r.timestamp));

        let mut user_map = HashMap::new();
        let mut item_map = HashMap::new();
        let mut idx_to_item = HashMap::new();
        let mut history_map: HashMap<usize, HashSet<usize>> = HashMap::new();
        
        let mut interactions = Vec::new();
        let mut test_set = Vec::new();

        let mut current_user_raw = u32::MAX;
        let mut user_buffer: Vec<(usize, usize)> = Vec::new();

        // Save the new idx for users and items
        for record in raw_data {
            let next_uid = user_map.len();
            let u_idx = *user_map.entry(record.uid).or_insert(next_uid);

            let i_idx = if let Some(&existing_idx) = item_map.get(&record.iid) {
                existing_idx
            } else {
                let new_idx = item_map.len();
                item_map.insert(record.iid, new_idx);
                idx_to_item.insert(new_idx, record.iid);
                new_idx
            };

            // Last item consumed considered Test set for each user
            if record.uid != current_user_raw {
                if !user_buffer.is_empty() {
                    if let Some(last) = user_buffer.pop() {
                        test_set.push(last);
                        interactions.append(&mut user_buffer);
                    }
                }
                current_user_raw = record.uid;
                user_buffer = Vec::new();
            }
            
            user_buffer.push((u_idx, i_idx));
            history_map.entry(u_idx).or_default().insert(i_idx);
        }
        
        if !user_buffer.is_empty() {
            if let Some(last) = user_buffer.pop() {
                test_set.push(last);
                interactions.append(&mut user_buffer);
            }
        }

        let movies_path = dir.join("movies.csv");
        let file_movies = File::open(movies_path)?;
        let mut rdr_movies = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(BufReader::new(file_movies));

        let mut movie_titles = HashMap::new();
        for result in rdr_movies.records() {
            let record = result?;
            let raw_iid: u32 = record[0].parse()?;
            if let Some(&internal_idx) = item_map.get(&raw_iid) {
                movie_titles.insert(internal_idx, record[1].to_string());
            }
        }

        let num_users = user_map.len();
        let num_items = item_map.len();
        let mut user_history = vec![HashSet::new(); num_users];
        for (u_idx, items) in history_map {
            user_history[u_idx] = items;
        }

        Ok(Self {
            num_users,
            num_items,
            interactions,
            test_set,
            user_history,
            movie_titles,
            item_raw_to_idx: item_map,
            idx_to_item_raw: idx_to_item,
            user_raw_to_idx: user_map,
        })
    }
    
    // Obtain the batch for the training
    pub fn get_batch(&self, batch_size: usize, device: &Device, rng: &mut impl Rng
    ) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
        let mut u_vec = Vec::with_capacity(batch_size);
        let mut i_vec = Vec::with_capacity(batch_size);
        let mut j_vec = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            // Random interactions
            let random_idx = rng.gen_range(0..self.interactions.len());
            // User and positive item batch
            let (u, i) = self.interactions[random_idx];

            // Random negative item batch
            let mut j = rng.gen_range(0..self.num_items);
            while self.user_history[u].contains(&j) {
                j = rng.gen_range(0..self.num_items);
            }

            u_vec.push(u as u32);
            i_vec.push(i as u32);
            j_vec.push(j as u32);
        }

        let u_tensor = Tensor::new(u_vec, device)?;
        let i_tensor = Tensor::new(i_vec, device)?;
        let j_tensor = Tensor::new(j_vec, device)?;

        Ok((u_tensor, i_tensor, j_tensor))
    }
}

// Sanity check
#[cfg(test)]
mod tests {
    use super::*;

    // Useful for the tests
    fn load_test_data() -> MovieLensData {
        MovieLensData::load("ml-latest-small").expect("Error loading the dataset.")
    }

    // Dimension of dataset
    #[test]
    fn test_1_dataset_dimensions() {
        let data = load_test_data();
        
        println!("---------------------------------------");
        println!("Sanity check 1: \nCorrect loading of the dataset.");
        println!("Users: {}, Items: {}, Interactions: {}", data.num_users, data.num_items, data.interactions.len());

        assert_eq!(data.num_users, 609, "Wrong number of user.");
        assert_eq!(data.num_items, 7363, "Wrong number of movies.");
        assert_eq!(data.interactions.len(), 61107, "Wrong number of interactions.");
        println!("---------------------------------------");
    }

    // Indexing of the items
    #[test]
    fn test_2_item_indexing() {
        let data = load_test_data();
        
        let test_cases = vec![
            (1, 124),
            (318, 351),
            (296, 336),
        ];
        
        println!("---------------------------------------");
        println!("Sanity check 2: \nCorrect indexing of the items.");
        
        for (raw_id, expected_idx) in test_cases {
            let actual_idx_option = data.item_raw_to_idx.get(&raw_id);
            let actual_idx = *actual_idx_option.unwrap();
            
            assert_eq!(actual_idx, expected_idx,
                "Mismatch for item, RawID {}: Rust idx {}, Python idx {}",
                raw_id, actual_idx, expected_idx
            );
            
            println!("OK item: RawID {} -> Index {} (Match confirmed)", raw_id, actual_idx);
        }
        println!("---------------------------------------");
    }

    // Indexing of the user
    #[test]
    fn test_3_user_indexing() {
        let data = load_test_data();
        
        let test_cases = vec![
            (1, 0),
            (100, 99),
            (300, 299),
        ];
        
        println!("---------------------------------------");
        println!("Sanity check 3: \nCorrect indexing of the users.");
        
        for (raw_id, expected_idx) in test_cases {
            let actual_idx_option = data.user_raw_to_idx.get(&raw_id);
            let actual_idx = *actual_idx_option.unwrap();
            
            assert_eq!(actual_idx, expected_idx,
                "Mismatch for movie RawID {}: Rust idx {}, Python idx {}",
                raw_id, actual_idx, expected_idx
            );
            
            println!("OK movie: RawID {} -> Index {} (Match confirmed)", raw_id, actual_idx);
        }
        println!("---------------------------------------");
    }

    // History of random users
    #[test]
    fn test_4_user_history() {
        let data = load_test_data();

        let test_cases = vec![
            (1, 200),
            (100, 135),
            (300, 29),
        ];

        println!("---------------------------------------");
        println!("Sanity check 4: \nHistory of random users.");

        for (raw_user_id, expected_len) in test_cases {
            if let Some(&idx) = data.user_raw_to_idx.get(&raw_user_id) {
                let history = &data.user_history[idx];
                let actual_len = history.len();

                assert_eq!(
                    actual_len,
                    expected_len,
                    "Mismatch for User {}: Rust found {} film, Python {}",
                    raw_user_id, actual_len, expected_len
                );

            } else {
                panic!("User {} exists in Python, but not in Rust", raw_user_id);
            }
        }
        println!("---------------------------------------");
    }
}