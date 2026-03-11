import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from tqdm import tqdm
from data import MovieLensData
from model import BPRModel

# constants used
EMBEDDING_DIM = 32
BATCH_SIZE = 1024
EPOCHS = 20
LR = 0.005
WD = 0.01

# Function to obtain deterministic output
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def sanity_check(data):
    print(f"\n------------------------------------------------")
    print("Sanity check 1: \nCorrect loading of the dataset.")
    print(f"Total users : {data.num_users}")
    print(f"Total movies: {data.num_items}")
    print(f"Interactions: {len(data.interactions)}")
    print(f"------------------------------------------------")

    print(f"------------------------------------------------")
    print("Sanity check 2: \nCorrect indexing of the items.")
    test_movie_ids = [1, 318, 296, 999999]  # 999999 not valid
    for raw_id in test_movie_ids:
        if raw_id in data.item_map:
            idx = data.item_map[raw_id]
            print(f"Raw ID {raw_id:<6} -> Index {idx}")
        else:
            print(f"Raw ID {raw_id:<6} -> Not found.")

    print(f"------------------------------------------------")

    print(f"------------------------------------------------")
    print("Sanity check 3: \nCorrect indexing of the users.")
    test_user_ids = [1, 100, 300]
    for raw_id in test_user_ids:
        if raw_id in data.user_map:
            idx = data.user_map[raw_id]
            print(f"Raw ID {raw_id:<6} -> Index {idx}")
        else:
            print(f"Raw ID {raw_id:<6} -> Not found.")
    print(f"------------------------------------------------")

    print(f"------------------------------------------------")
    print("Sanity check 4: \nHistory of random users.")
    for raw_id in test_user_ids:
        if raw_id in data.user_map:
            idx = data.user_map[raw_id]
            history_len = len(data.user_history.get(idx, []))
            print(f"User Raw {raw_id:<3} (Idx {idx:<3}), history of {history_len} movies.")
    print(f"------------------------------------------------")

# Hit ratio function
def calculate_hit_ratio(model, data, device, k=10):
    model.eval()
    hits = 0
    total_users = len(data.test_df)

    print(f"\nEvaluation on {total_users} users (Test Set)...")

    # Get the user and item idx
    test_pairs = data.test_df[['u_idx', 'i_idx']].values

    with torch.no_grad():
        for u_idx, true_item_idx in tqdm(test_pairs, desc="Evaluating"):
            u_tensor = torch.tensor([u_idx], dtype=torch.long, device=device)

            all_scores = model(u_tensor)

            # Save the computed score for the item
            true_item_score = all_scores[true_item_idx].item()

            # Mask with -inf all the seen items
            seen_items = data.user_history.get(u_idx, set())
            seen_items_list = list(seen_items)
            all_scores[seen_items_list] = -float('inf')

            # The only item not masked is the test item (remember that the user saw it)
            all_scores[true_item_idx] = true_item_score

            # Check if the test item is in top k
            _, top_indices = torch.topk(all_scores, k)
            top_indices = top_indices.cpu().numpy()

            if true_item_idx in top_indices:
                hits += 1

    hr = hits / total_users
    return hr

# Get avg, p50, p90, p95, p99 inference time
def benchmark_latency(model, data, device, num_requests=1000):
    model.eval()
    latencies = []

    # Random users
    test_users = np.random.randint(0, data.num_users, size=num_requests)

    print(f"\nStarting benchmark on {num_requests} requests")

    with torch.no_grad():
        for u_idx in test_users:
            u_tensor = torch.tensor([u_idx], dtype=torch.long, device=device)
            start = time.perf_counter()

            _ = model(u_tensor).cpu()

            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()

            # Latencies in ms
            latencies.append((end - start) * 1000)

    # All the statistics
    latencies = np.array(latencies)
    print(f"------------------------------------------------")
    print("Python Inference Stats")
    print(f"Avg: {np.mean(latencies):.4f} ms")
    print(f"P50: {np.percentile(latencies, 50):.4f} ms")
    print(f"P90: {np.percentile(latencies, 90):.4f} ms")
    print(f"P95: {np.percentile(latencies, 95):.4f} ms")
    print(f"P99: {np.percentile(latencies, 99):.4f} ms")
    print(f"------------------------------------------------")

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = MovieLensData("ml-1m-csv")
    sanity_check(data)


    model = BPRModel(data.num_users, data.num_items, EMBEDDING_DIM).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    # Training Loop
    model.train()
    steps_per_epoch = len(data.interactions) // BATCH_SIZE

    # sync between GPU and CPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Start the time
    start_time = time.time()

    print("Training:")
    for epoch in range(EPOCHS):
        total_loss = 0.0

        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}")

        for _ in pbar:
            # Get Batch
            u, i, j = data.get_batch(BATCH_SIZE, device)

            # Forward
            optimizer.zero_grad()
            diff = model.calculate_loss(u, i, j)

            # Loss = -ln(sigmoid(diff)) = softplus(-diff)
            loss = F.softplus(-diff).mean()

            # Backward
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        print(f"Epoch {epoch + 1} Avg Loss: {total_loss / steps_per_epoch:.4f}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # save the time
    end_time = time.time()
    training_time = end_time - start_time

    hit_ratio = calculate_hit_ratio(model, data, device, k=10)
    print(f"------------------------------------------------")
    print(f"Hit Ratio @ 10: {hit_ratio:.4f} ({hit_ratio * 100:.2f}%)")
    print(f"------------------------------------------------")

    # Inference
    test_user_idx = 100

    all_seen_ids = data.user_history[test_user_idx]
    all_seen_titles = [data.movie_titles.get(idx, 'Unknown') for idx in all_seen_ids]

    # Alphabetical order
    all_seen_titles.sort()

    print(f"\n------------------------------------------------")
    print(f"First 10 items seen by a random user (idx: {test_user_idx})")
    print(f"------------------------------------------------")
    for title in all_seen_titles[:10]:
        print(f" - {title}")

    # Item recommendations
    user_input_tensor = torch.tensor([test_user_idx], dtype=torch.long, device=device)

    # sync between GPU and CPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Start the time
    start_time = time.perf_counter()
    with torch.no_grad():
        scores = model(user_input_tensor)

    scores_np = scores.cpu().numpy()

    # save the time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    inference_time = (end_time - start_time) * 1000

    seen_mask = list(data.user_history[test_user_idx])
    scores_np[seen_mask] = -np.inf

    top_k_indices = np.argsort(scores_np)[-10:][::-1]

    print(f"------------------------------------------------")
    print(f"Example of recommendations for a random user (idx {test_user_idx})")
    for rank, idx in enumerate(top_k_indices):
        title = data.movie_titles.get(idx, "Unknown")
        score = scores_np[idx]
        print(f"{rank + 1}. {title} (Score: {score:.2f})")

    print(f"Training time: {training_time:.2f}s")
    print(f"Inference time: {inference_time:.2f}ms")

    benchmark_latency(model, data, device)

    print(f"------------------------------------------------")
    print("\nExport the model")
    model.to("cpu")
    model.eval()

    dummy_input = torch.tensor([0], dtype=torch.long)

    torch.onnx.export(
        model,
        (dummy_input,),
        "bpr_model_1m.onnx",
        input_names=['user_id'],
        output_names=['scores'],
        dynamic_axes={
            'user_id': {0: 'batch_size'},
            'scores': {0: 'batch_size'}
        }
    )

main()
