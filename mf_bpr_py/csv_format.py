import pandas as pd
import os

# Load ratings
print("Reading ratings.dat...")
df = pd.read_csv(
    "ml-1m/ratings.dat",
    sep="::",
    names=["userId", "movieId", "rating", "timestamp"],
    engine="python"
)

# Load movies
print("Reading movies.dat...")
movies = pd.read_csv(
    "ml-1m/movies.dat",
    sep="::",
    names=["movieId", "title", "genres"],
    engine="python",
    encoding="latin-1"
)

# Save in csv format
output_folder = "ml-1m-csv"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

df.to_csv(f"{output_folder}/ratings.csv", index=False)
movies.to_csv(f"{output_folder}/movies.csv", index=False)

print("Done.")