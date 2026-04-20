import argparse
import os

import pandas as pd
import numpy as np


def split_one_class(
    class_df: pd.DataFrame,
    train_ratio: float,
    seed: int,
):
    n = len(class_df)
    if n < 2:
        raise ValueError("Each class must have at least 2 samples")

    n_train = int(round(n * train_ratio))
    n_train = max(1, min(n_train, n - 1))
    n_test = n - n_train

    if n_train <= 0 or n_test <= 0:
        raise ValueError(
            f"Invalid split sizes for class with {n} samples: "
            f"train={n_train}, test={n_test}"
        )

    shuffled = class_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train_df = shuffled.iloc[:n_train]
    test_df = shuffled.iloc[n_train:]
    return train_df, test_df


def main():
    parser = argparse.ArgumentParser(description="Split keypoint dataset into train/test")
    parser.add_argument("--in", dest="inp", default=os.path.join("output", "dataset_normalized.csv"))
    parser.add_argument("--out-dir", default="output")
    parser.add_argument("--train", type=float, default=0.80)
    parser.add_argument("--test", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not np.isclose(args.train + args.test, 1.0, atol=1e-6):
        raise ValueError("train + test must be 1.0")

    df = pd.read_csv(args.inp)
    if "label" not in df.columns:
        raise ValueError("Input CSV must contain a 'label' column")

    labels = sorted(df["label"].unique().tolist())
    train_parts = []
    test_parts = []

    for i, label in enumerate(labels):
        class_df = df[df["label"] == label]
        train_c, test_c = split_one_class(
            class_df=class_df,
            train_ratio=args.train,
            seed=args.seed + i,
        )
        train_parts.append(train_c)
        test_parts.append(test_c)

    train_df = pd.concat(train_parts, axis=0).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    test_df = pd.concat(test_parts, axis=0).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    os.makedirs(args.out_dir, exist_ok=True)
    train_out = os.path.join(args.out_dir, "train.csv")
    test_out = os.path.join(args.out_dir, "test.csv")

    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    print(f"Saved: {train_out}")
    print(f"Saved: {test_out}")
    print(f"train={len(train_df)}, test={len(test_df)}")
    print("Train per label:")
    print(train_df["label"].value_counts().sort_index())
    print("Test per label:")
    print(test_df["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
