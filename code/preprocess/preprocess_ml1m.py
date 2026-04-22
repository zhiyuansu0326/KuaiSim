import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert MovieLens-1M raw files into KuaiKuaiSim MLSeqReader format."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="dataset/ml-1m",
        help="Directory containing ratings.dat, users.dat, movies.dat",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset/ml-1m",
        help="Directory to write processed csv files",
    )
    parser.add_argument(
        "--neg_ratio",
        type=float,
        default=1.0,
        help="Number of negative samples per positive record (user-level sampling)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for negative sampling",
    )
    return parser.parse_args()


def load_raw(input_dir: Path):
    ratings = pd.read_csv(
        input_dir / "ratings.dat",
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    users = pd.read_csv(
        input_dir / "users.dat",
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        encoding="latin-1",
    )
    movies = pd.read_csv(
        input_dir / "movies.dat",
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    return ratings, users, movies


def build_positive_log(ratings: pd.DataFrame) -> pd.DataFrame:
    pos = ratings.copy()
    pos["is_click"] = 1
    pos["is_like"] = (pos["rating"] >= 4).astype(int)
    pos["is_star"] = (pos["rating"] >= 5).astype(int)
    return pos


def sample_negative_log(pos: pd.DataFrame, neg_ratio: float, seed: int) -> pd.DataFrame:
    if neg_ratio <= 0:
        return pos.iloc[0:0].copy()

    rng = np.random.default_rng(seed)
    all_movie_ids = pos["movie_id"].drop_duplicates().to_numpy()
    user_groups = pos.groupby("user_id", sort=False)

    neg_rows = []
    for uid, group in tqdm(user_groups, desc="Negative sampling"):
        n_pos = len(group)
        n_neg = int(np.floor(n_pos * neg_ratio))
        if n_neg <= 0:
            continue

        user_pos_set = set(group["movie_id"].tolist())
        candidates = [mid for mid in all_movie_ids if mid not in user_pos_set]
        if len(candidates) == 0:
            continue

        replace = n_neg > len(candidates)
        sampled_movies = rng.choice(candidates, size=n_neg, replace=replace)
        base_ts = rng.choice(group["timestamp"].to_numpy(), size=n_neg, replace=True)
        # Shift negative timestamps slightly to keep deterministic ordering within user.
        ts_shift = rng.integers(1, 61, size=n_neg)
        sampled_ts = base_ts + ts_shift

        neg_part = pd.DataFrame(
            {
                "user_id": uid,
                "movie_id": sampled_movies.astype(np.int64),
                "rating": 0,
                "timestamp": sampled_ts.astype(np.int64),
                "is_click": 0,
                "is_like": 0,
                "is_star": 0,
            }
        )
        neg_rows.append(neg_part)

    if not neg_rows:
        return pos.iloc[0:0].copy()
    return pd.concat(neg_rows, ignore_index=True)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ratings, users, movies = load_raw(input_dir)

    users_processed = users[["user_id", "gender", "age"]].copy()
    movies_processed = movies[["movie_id", "genres"]].copy()
    movies_processed["genres"] = movies_processed["genres"].astype(str).str.replace("|", ",", regex=False)

    pos_log = build_positive_log(ratings)
    neg_log = sample_negative_log(pos_log, args.neg_ratio, args.seed)
    log = pd.concat([pos_log, neg_log], ignore_index=True)
    log = log.sort_values(["user_id", "timestamp", "movie_id"], kind="mergesort").reset_index(drop=True)
    log = log[["user_id", "movie_id", "timestamp", "is_click", "is_like", "is_star"]]

    users_processed.to_csv(output_dir / "users_processed.csv", index=False)
    movies_processed.to_csv(output_dir / "movies_processed.csv", index=False)
    log.to_csv(output_dir / "log_session.csv", index=False)

    print("Saved files:")
    print(output_dir / "users_processed.csv")
    print(output_dir / "movies_processed.csv")
    print(output_dir / "log_session.csv")
    print("Log stats:")
    print(log[["is_click", "is_like", "is_star"]].mean().to_dict())
    print("Rows:", len(log), "Users:", log["user_id"].nunique(), "Items:", log["movie_id"].nunique())


if __name__ == "__main__":
    main()
