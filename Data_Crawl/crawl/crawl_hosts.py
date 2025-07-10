import pandas as pd
import ast

# Read the CSV file
df = pd.read_csv(r"C:\Users\fraxi\OneDrive\Desktop\code task\Amsterdam\listings.csv")

# Select unique hosts and extract relevant columns
host_df = df.drop_duplicates(subset="host_id")[
    [
        "host_id",
        "host_since",
        "host_about",
        "host_verifications",
        "host_is_superhost",
        "host_total_listings_count",
        "number_of_reviews",
        "review_scores_rating",
        "host_response_rate",
        "host_acceptance_rate"
    ]
].copy()

# Extract the year the host joined
host_df["host_join_year"] = pd.to_datetime(host_df["host_since"], errors='coerce').dt.year

# Safely count the number of verification methods
def count_verified_sources(x):
    try:
        return len(ast.literal_eval(x)) if pd.notna(x) else 0
    except:
        return 0

host_df["num_verified_sources"] = host_df["host_verifications"].apply(count_verified_sources)

# Convert superhost status to boolean
host_df["is_superhost"] = host_df["host_is_superhost"] == "t"

# Convert response rate and acceptance rate to numeric values
host_df["host_response_rate"] = host_df["host_response_rate"].str.rstrip('%')
host_df["host_acceptance_rate"] = host_df["host_acceptance_rate"].str.rstrip('%')

# Build the final output DataFrame
final_host_df = host_df[
    [
        "host_id",
        "host_join_year",
        "host_about",
        "num_verified_sources",
        "is_superhost",
        "host_total_listings_count",
        "number_of_reviews",
        "review_scores_rating",
        "host_response_rate",
        "host_acceptance_rate"
    ]
]

# Optional: Save to local CSV
final_host_df.to_csv(r"C:\Users\fraxi\OneDrive\Desktop\code task\Amsterdam\amsterdam_host_profiles.csv", index=False)

# Optional: Print the first few rows
print(final_host_df.head())
