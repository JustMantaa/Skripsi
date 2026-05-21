import argparse
from pathlib import Path

import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]


def split_one_class(
    class_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
):
    n = len(class_df)
    if n < 3:
        raise ValueError("Each class must have at least 3 samples for train/val/test split")

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    # Pastikan tiap split minimal 1 sample
    if n_train <= 0:
        n_train = 1
    if n_val <= 0:
        n_val = 1
    n_test = n - n_train - n_val

    if n_test <= 0:
        # Kurangi split terbesar yang masih >1 hingga test punya minimal 1 sample
        while n_test <= 0:
            if n_train >= n_val and n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                raise ValueError(
                    f"Tidak bisa membagi class dengan {n} sample ke train/val/test secara valid"
                )
            n_test = n - n_train - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"Invalid split sizes for class with {n} samples: "
            f"train={n_train}, val={n_val}, test={n_test}"
        )

    shuffled = class_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train_df = shuffled.iloc[:n_train]
    val_df = shuffled.iloc[n_train:n_train + n_val]
    test_df = shuffled.iloc[n_train + n_val:]
    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description="Split keypoint dataset into train/val/test")
    parser.add_argument("--in", dest="inp", default=str(BASE_DIR / "output" / "dataset_normalized.csv"))
    parser.add_argument("--out-dir", default=str(BASE_DIR / "output"))
    parser.add_argument("--train", type=float, default=0.80)
    parser.add_argument("--val", type=float, default=0.10)
    parser.add_argument("--test", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not np.isclose(args.train + args.val + args.test, 1.0, atol=1e-6):
        raise ValueError("train + val + test must be 1.0")

    input_path = Path(args.inp)
    out_dir = Path(args.out_dir)

    if not input_path.is_absolute():
        input_path = BASE_DIR / input_path
    if not out_dir.is_absolute():
        out_dir = BASE_DIR / out_dir

    df = pd.read_csv(input_path)
    if "label" not in df.columns:
        raise ValueError("Input CSV must contain a 'label' column")

    labels = sorted(df["label"].unique().tolist())
    train_parts = []
    val_parts = []
    test_parts = []

    for i, label in enumerate(labels):
        class_df = df[df["label"] == label]
        train_c, val_c, test_c = split_one_class(
            class_df=class_df,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            seed=args.seed + i,
        )
        train_parts.append(train_c)
        val_parts.append(val_c)
        test_parts.append(test_c)

    train_df = pd.concat(train_parts, axis=0).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    val_df = pd.concat(val_parts, axis=0).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    test_df = pd.concat(test_parts, axis=0).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_out = out_dir / "train.csv"
    val_out = out_dir / "val.csv"
    test_out = out_dir / "test.csv"

    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    test_df.to_csv(test_out, index=False)

    print(f"Saved: {train_out}")
    print(f"Saved: {val_out}")
    print(f"Saved: {test_out}")
    print(f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print("Train per label:")
    print(train_df["label"].value_counts().sort_index())
    print("Val per label:")
    print(val_df["label"].value_counts().sort_index())
    print("Test per label:")
    print(test_df["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
