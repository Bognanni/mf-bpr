use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use rand::Rng;
use candle_core::{Device, Tensor};
use serde::Deserialize;

// generic struct to save the raw interactions
struct RawInteraction {
    uid: String,
    iid: String,
    timestamp: u64,
}

// struct to deserialize the jsonl Amazon file
#[derive(Deserialize)]
struct AmazonReview {
    user_id: String,
    parent_asin: String,
    rating: f32,
    timestamp: u64,
}

// struct to deserialize meta data of Amazon All Beauty
#[derive(Deserialize)]
struct AmazonMeta {
    parent_asin: String,
    title: Option<String>, // Option if there is not a title
}

// universal struct to handle the dataset preprocessed
pub struct RecSysData {
    pub num_users: usize,
    pub num_items: usize,
    pub interactions: Vec<(usize, usize)>,
    pub user_history: Vec<HashSet<usize>>,
    pub test_set: Vec<(usize, usize)>,
    pub item_titles: HashMap<usize, String>,
    pub item_raw_to_idx: HashMap<String, usize>, // String to support Amazon All Beauty
    pub user_raw_to_idx: HashMap<String, usize>,
    pub idx_to_item_raw: HashMap<usize, String>
}
impl RecSysData {
    // load the movielens dataset
    pub fn load_movielens<P: AsRef<Path>>(dir_path: P) -> anyhow::Result<Self> {
        println!("Loading MovieLens dataset from {:?}.", dir_path.as_ref());
        let dir = dir_path.as_ref();
        let file = File::open(dir.join("ratings.csv"))?;
        let mut rdr = csv::ReaderBuilder::new().has_headers(true).from_reader(BufReader::new(file));

        let mut raw_data = Vec::new();
        // for each record (row)
        for result in rdr.records() {
            let record = result?;
            let rating: f32 = record[2].parse()?;
            // consider only reviews with rating >= 3.5
            if rating >= 3.5 {
                // save the raw interaction
                raw_data.push(RawInteraction {
                    uid: record[0].to_string(), // cast into String
                    iid: record[1].to_string(), // cast into String
                    timestamp: record[3].parse()?,
                });
            }
        }

        // retrieve titles
        let mut titles = HashMap::new();
        if let Ok(file_movies) = File::open(dir.join("movies.csv")) {
            let mut rdr_movies = csv::ReaderBuilder::new().has_headers(true).from_reader(BufReader::new(file_movies));
            for result in rdr_movies.records() {
                if let Ok(record) = result {
                    titles.insert(record[0].to_string(), record[1].to_string());
                }
            }
        }
        
        // return the builded dataset from raw data
        Self::build_dataset(raw_data, titles)
    }

    // load the Amazon All Beauty
    pub fn load_amazon<P: AsRef<Path>, M: AsRef<Path>>(jsonl_path: P, meta_jsonl_path: Option<M>) -> anyhow::Result<Self> {
        println!("Loading Amazon All Beauty from {:?}.", jsonl_path.as_ref());
        let dir = jsonl_path.as_ref();
        let file = File::open(dir.join("All_Beauty.jsonl"))?;
        let reader = BufReader::new(file);

        let mut raw_data = Vec::new();
        // HashSet for the valid asins (rating >= 3.5)
        let mut valid_asins = HashSet::new();

        // for each line (review)
        for line in reader.lines() {
            let line = line?;
            // deserialize data
            let review: AmazonReview = serde_json::from_str(&line)?;
            
            // same logic of before (consider only ratings >= 3.5)
            if review.rating >= 3.5 {
                valid_asins.insert(review.parent_asin.clone());
                // save the raw data
                raw_data.push(RawInteraction {
                    uid: review.user_id,
                    iid: review.parent_asin,
                    timestamp: review.timestamp,
                });
            }
        }

        // retrieve titles
        let mut titles = HashMap::new();

        if let Some(meta_path) = meta_jsonl_path {
            println!("Extract items' title from metadata file : {:?}.", meta_path.as_ref());
            let dir = meta_path.as_ref();
            let meta_file = File::open(dir.join("meta_All_Beauty.jsonl"))?;
            let meta_reader = BufReader::new(meta_file);
            
            for line in meta_reader.lines() {
                let line = line?;
                // consider only the complete lines
                if let Ok(meta_data) = serde_json::from_str::<AmazonMeta>(&line) {
                    // save only the titles from the filtered items
                    if valid_asins.contains(&meta_data.parent_asin) {
                        let title_str = meta_data.title.unwrap_or_else(|| format!("Missing title (ASIN: {})", meta_data.parent_asin));
                        titles.insert(meta_data.parent_asin, title_str);
                    }
                }
            }
        }

        // build dataseg from raw data and titles
        Self::build_dataset(raw_data, titles)
    }

    // function shared between datasets to build them
    fn build_dataset(mut raw_data: Vec<RawInteraction>, raw_titles: HashMap<String, String>) -> anyhow::Result<Self> {
        // sort by user and by time
        raw_data.sort_by(|a, b| a.uid.cmp(&b.uid).then(a.timestamp.cmp(&b.timestamp)));

        // all the variables considered for a dataset
        let mut user_map = HashMap::new();
        let mut item_map = HashMap::new();
        let mut idx_to_item = HashMap::new();
        let mut history_map: HashMap<usize, HashSet<usize>> = HashMap::new();
        
        let mut interactions = Vec::new();
        let mut test_set = Vec::new();

        let mut current_user_raw = String::new();
        let mut user_buffer: Vec<(usize, usize)> = Vec::new();

        // for each interaction
        for record in raw_data {
            let next_uid = user_map.len();
            // get the user idx that can be already saved in user_map associated to the asin saved before
            // or a new value equals to the current len o user map + 1
            let u_idx = *user_map.entry(record.uid.clone()).or_insert(next_uid);

            // item idx already existing got from the item_map or added to the map and equals to the
            // current len of the map
            let i_idx = if let Some(&existing_idx) = item_map.get(&record.iid) {
                existing_idx
            } else {
                let new_idx = item_map.len();
                item_map.insert(record.iid.clone(), new_idx);
                idx_to_item.insert(new_idx, record.iid.clone());
                new_idx
            };

            // if the history user is finished (remember, it was sorted before)
            if record.uid != current_user_raw {
                // if the user has an history
                if !user_buffer.is_empty() {
                    // get the last item and use it as a test (leave one out)
                    if let Some(last) = user_buffer.pop() {
                        test_set.push(last);
                        // the remaining items go in interactions and they'll be used for training
                        interactions.append(&mut user_buffer);
                    }
                }
                current_user_raw = record.uid;
                // empty buffer for the next new user
                user_buffer = Vec::new();
            }
            
            // buffer with the interactions of the current user
            user_buffer.push((u_idx, i_idx));
            history_map.entry(u_idx).or_default().insert(i_idx);
        }
        
        // for the last user we do the same, put the last item in test_set and the others in interactions
        if let Some(last) = user_buffer.pop() {
            test_set.push(last);
            interactions.append(&mut user_buffer);
        }

        let num_users = user_map.len();
        let num_items = item_map.len();
        // a vec of HashSet, len equals to the num of users and each one contains the user history
        let mut user_history = vec![HashSet::new(); num_users];
        for (u_idx, items) in history_map {
            user_history[u_idx] = items;
        }

        // save also the item titles associating the new idx with the title
        let mut item_titles = HashMap::new();
        for (idx, raw_id) in &idx_to_item {
            let title = raw_titles.get(raw_id).cloned().unwrap_or_else(|| format!("ASIN: {}", raw_id));
            item_titles.insert(*idx, title);
        }

        println!("Dataset loaded! Users: {}, Items: {}, Training interactions: {}", num_users, num_items, interactions.len());

        Ok(Self {
            num_users,
            num_items,
            interactions,
            test_set,
            user_history,
            item_titles,
            item_raw_to_idx: item_map,
            idx_to_item_raw: idx_to_item,
            user_raw_to_idx: user_map,
        })
    }

    // function to get the batches for training
    pub fn get_batch(&self, batch_size: usize, device: &Device, rng: &mut impl Rng) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
        let mut u_vec = Vec::with_capacity(batch_size);
        let mut i_vec = Vec::with_capacity(batch_size);
        let mut j_vec = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            let random_idx = rng.gen_range(0..self.interactions.len());
            let (u, i) = self.interactions[random_idx];

            let mut j = rng.gen_range(0..self.num_items);
            while self.user_history[u].contains(&j) {
                j = rng.gen_range(0..self.num_items);
            }

            u_vec.push(u as u32);
            i_vec.push(i as u32);
            j_vec.push(j as u32);
        }

        Ok((Tensor::new(u_vec, device)?, Tensor::new(i_vec, device)?, Tensor::new(j_vec, device)?))
    }
}

// Generalized sanity checks
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;
    use std::env;

    // global shared space and thread-safe (Singleton)
    static SHARED_DATA: OnceLock<RecSysData> = OnceLock::new();

    // load the data set one time and then shared between the tests
    fn get_test_data() -> &'static RecSysData {
        SHARED_DATA.get_or_init(|| {
            // read the env variable, else movielens as default
            let dataset_type = env::var("TEST_DATASET").unwrap_or_else(|_| "movielens".to_string());
            
            println!("Starting tests on : {} dataset.", dataset_type.to_uppercase());

            match dataset_type.as_str() {
                "amazon" => RecSysData::load_amazon("All_Beauty.jsonl", Some("meta_All_Beauty.jsonl"))
                    .expect("Error: not possible load the dataset."),
                "movielens" => RecSysData::load_movielens("ml-1m-csv")
                    .expect("Error: not possible load the dataset."),
                _ => panic!("Dataset not supported. Use 'movielens' or 'amazon'."),
            }
        })
    }

    #[test]
    fn test_1_dataset_dimensions() {
        // data shared
        let data = get_test_data();
        let dataset_type = env::var("TEST_DATASET").unwrap_or_else(|_| "movielens".to_string());
        
        println!("---------------------------------------");
        println!("Sanity check 1: \nCorrect loading of the dataset.");
        
        println!("Users: {}, Items: {}, Interactions: {}", data.num_users, data.num_items, data.interactions.len());
        let (expected_users, expected_items, expected_interactions) = match dataset_type.as_str() {
            "movielens" => (6038, 3533, 569243),
            "amazon" => (455586, 91187, 44521),
            _ => panic!("Unknown dataset."),
        };

        assert_eq!(data.num_users, expected_users, "Wrong number of users.");
        assert_eq!(data.num_items, expected_items, "Wrong number of items.");
        assert_eq!(data.interactions.len(), expected_interactions, "Wrong number of interactions.");
        
        println!("---------------------------------------");
    }

    #[test]
    fn test_2_item_indexing() {
        let data = get_test_data();
        let dataset_type = env::var("TEST_DATASET").unwrap_or_else(|_| "movielens".to_string());
        
        let test_cases = match dataset_type.as_str() {
            "movielens" => vec![
                ("3186".to_string(), 0),
                ("1270".to_string(), 1),
                ("1721".to_string(), 2),
            ],
            "amazon" => vec![
                ("B013HR1A92".to_string(), 0),
                ("B0BTT658PQ".to_string(), 1),
                ("B00PBDMRES".to_string(), 2),
            ],
            _ => panic!("Unknown dataset."),
        };
        
        println!("---------------------------------------");
        println!("Sanity check 2: \nCorrect indexing of the items.");
        
        for (raw_id, expected_idx) in test_cases {
            // Usiamo unwrap_or_else per avere un messaggio di errore chiaro se l'ID non esiste
            let actual_idx = *data.item_raw_to_idx.get(&raw_id).unwrap_or_else(|| {
                panic!("The RawID item '{}' does not exist in the map!", raw_id)
            });
            
            assert_eq!(actual_idx, expected_idx, "Mismatch for item, RawID {}: Rust idx {}, Python idx {}",raw_id, actual_idx, expected_idx);
            
            println!("OK item: RawID {:<15} -> Index {} (Match confirmed)", raw_id, actual_idx);
        }
        println!("---------------------------------------");
    }

    #[test]
    fn test_3_user_indexing() {
        let data = get_test_data();
        let dataset_type = env::var("TEST_DATASET").unwrap_or_else(|_| "movielens".to_string());
        
        let test_cases = match dataset_type.as_str() {
            "movielens" => vec![
                ("1".to_string(), 0),
                ("2".to_string(), 1),
                ("3".to_string(), 2),
            ],
            "amazon" => vec![
                ("AE222BBOVZIF42YOOPNBXL4UUMYA".to_string(), 0),
                ("AE222FP7YRNFCEQ2W3ZDIGMSYTLQ".to_string(), 1),
                ("AE222X475JC6ONXMIKZDFGQ7IAUA".to_string(), 2),
            ],
            _ => panic!("Unkwown dataset."),
        };
        
        println!("---------------------------------------");
        println!("Sanity check 3: \nCorrect indexing of the users.");
        
        for (raw_id, expected_idx) in test_cases {
            let actual_idx = *data.user_raw_to_idx.get(&raw_id).unwrap_or_else(|| {
                panic!("The RawID user '{}' does not exist in the map!", raw_id)
            });
            
            assert_eq!(actual_idx, expected_idx, "Mismatch for user RawID {}: Rust idx {}, Python idx {}", raw_id, actual_idx, expected_idx);
            
            println!("OK user: RawID {:<15} -> Index {} (Match confirmed)", raw_id, actual_idx);
        }
        println!("---------------------------------------");
    }

    #[test]
    fn test_4_user_history() {
        let data = get_test_data();
        let dataset_type = env::var("TEST_DATASET").unwrap_or_else(|_| "movielens".to_string());

        let test_cases = match dataset_type.as_str() {
            "movielens" => vec![
                ("1", 45),
                ("2", 73),
                ("3", 37),
            ],
            "amazon" => vec![
                ("AE222BBOVZIF42YOOPNBXL4UUMYA", 1),
                ("AE222FP7YRNFCEQ2W3ZDIGMSYTLQ", 1),
                ("AE222X475JC6ONXMIKZDFGQ7IAUA", 1),
            ],
            _ => panic!("Unkwnown dataset."),
        };

        println!("---------------------------------------");
        println!("History of the first three users.");

        for (raw_user_id, expected_len) in test_cases {
            let idx = *data.user_raw_to_idx.get(raw_user_id).unwrap_or_else(|| {
                panic!("The RawID user '{}' does not exist in the map!", raw_user_id)
            });
            
            assert!(idx < data.user_history.len(), "There is no history for the user {}!", raw_user_id);
            
            let actual_len = data.user_history[idx].len();

            assert_eq!(actual_len, expected_len,
                "Mismatch for User {}: Rust found {} items, Python {}.",
                raw_user_id, actual_len, expected_len
            );
        }
        println!("---------------------------------------");
    }
}