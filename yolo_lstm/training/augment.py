import argparse
import os

import numpy as np
import pandas as pd


def temporal_resample(sequence: np.ndarray, speed_factor: float) -> np.ndarray:
    timesteps, features = sequence.shape
    src_t = np.linspace(0.0, timesteps - 1, timesteps, dtype=np.float32)
    warped_t = np.linspace(0.0, timesteps - 1, timesteps, dtype=np.float32)
    warped_t = warped_t / speed_factor
    warped_t = np.clip(warped_t, 0.0, timesteps - 1)

    out = np.empty_like(sequence, dtype=np.float32)
    for i in range(features):
        out[:, i] = np.interp(warped_t, src_t, sequence[:, i]).astype(np.float32)
    return out


def temporal_shift_edge(sequence: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0:
        return sequence
    t, f = sequence.shape
    out = np.empty_like(sequence, dtype=np.float32)
    if shift > 0:
        out[:shift] = sequence[0]
        out[shift:] = sequence[: t - shift]
    else:
        k = -shift
        out[t - k :] = sequence[-1]
        out[: t - k] = sequence[k:]
    return out


def augment_sequence(seq: np.ndarray, landmarks: int, rng: np.random.Generator) -> np.ndarray:
    t, f = seq.shape
    seq3 = seq.reshape(t, landmarks, 3).copy().astype(np.float32)

    # Temporal transforms: speed + bounded shift to vary dynamics safely.
    speed = float(rng.uniform(0.85, 1.15))
    seq3 = temporal_resample(seq3.reshape(t, f), speed).reshape(t, landmarks, 3)
    shift_t = int(rng.integers(-3, 4))
    seq3 = temporal_shift_edge(seq3.reshape(t, f), shift_t).reshape(t, landmarks, 3)

    # Spatial transforms: anisotropic scaling, rotation, translation, drift.
    scale_xy = rng.uniform(0.90, 1.10, size=(1, 1, 2)).astype(np.float32)
    rot_deg = float(rng.uniform(-12.0, 12.0))
    rot_rad = np.deg2rad(rot_deg)
    rot = np.array(
        [[np.cos(rot_rad), -np.sin(rot_rad)], [np.sin(rot_rad), np.cos(rot_rad)]],
        dtype=np.float32,
    )
    shift_xy = rng.uniform(-0.04, 0.04, size=(1, 1, 2)).astype(np.float32)
    start_drift = rng.uniform(-0.02, 0.02, size=(1, 1, 2)).astype(np.float32)
    end_drift = rng.uniform(-0.02, 0.02, size=(1, 1, 2)).astype(np.float32)
    z_scale = float(rng.uniform(0.90, 1.10))

    center = seq3[:, :, :2].mean(axis=(0, 1), keepdims=True)
    xy = (seq3[:, :, :2] - center) * scale_xy
    xy = np.einsum("tld,dc->tlc", xy, rot)
    alphas = np.linspace(0.0, 1.0, t, dtype=np.float32).reshape(t, 1, 1)
    drift = start_drift * (1.0 - alphas) + end_drift * alphas
    seq3[:, :, :2] = xy + center + shift_xy + drift
    seq3[:, :, 2] = seq3[:, :, 2] * z_scale

    # Tracking-noise simulation with time-varying magnitude.
    noise_scale = rng.uniform(0.002, 0.009, size=(t, 1, 1)).astype(np.float32)
    noise = rng.normal(0.0, 1.0, size=seq3.shape).astype(np.float32) * noise_scale
    seq3 += noise

    # Local landmark disturbance for robustness (small subset only).
    if rng.random() < 0.35:
        k = int(rng.integers(1, min(4, landmarks + 1)))
        idx = rng.choice(landmarks, size=k, replace=False)
        lm_noise = rng.normal(0.0, 0.012, size=(t, k, 3)).astype(np.float32)
        seq3[:, idx, :] += lm_noise

    # Frame drop simulation: replace a few frames with neighbor frame.
    if t > 4 and rng.random() < 0.25:
        n_drop = max(1, int(0.1 * t))
        drop_idx = np.sort(rng.choice(t, size=n_drop, replace=False))
        for di in drop_idx:
            src = max(di - 1, 0)
            seq3[di] = seq3[src]

    seq3[:, :, :2] = np.clip(seq3[:, :, :2], 0.0, 1.0)
    seq3[:, :, 2] = np.clip(seq3[:, :, 2], -1.0, 1.0)

    return seq3.reshape(t, f).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Augment training split only")
    parser.add_argument("--in", dest="inp", default=os.path.join("output", "train.csv"))
    parser.add_argument("--out", default=os.path.join("output", "train_aug.csv"))
    parser.add_argument("--aug-per-sample", type=int, default=3)
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=0,
        help="Final number of samples per label after augmentation. Set <=0 to disable cap and use aug-per-sample mode.",
    )
    parser.add_argument("--landmarks", type=int, default=21)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.inp)
    if "label" not in df.columns:
        raise ValueError("Input CSV must contain a 'label' column")

    feature_cols = [c for c in df.columns if c != "label"]
    X_train = df[feature_cols].to_numpy(dtype=np.float32)
    y_train = df["label"].to_numpy()

    if X_train.shape[1] % (args.landmarks * 3) != 0:
        raise ValueError("Feature count is incompatible with landmarks*3")

    timesteps = X_train.shape[1] // (args.landmarks * 3)
    X_train = X_train.reshape(-1, timesteps, args.landmarks * 3)

    rng = np.random.default_rng(args.seed)

    if args.target_per_class > 0:
        labels = sorted(pd.unique(y_train).tolist())
        x_parts = []
        y_parts = []

        for label in labels:
            class_mask = y_train == label
            x_class = X_train[class_mask]
            y_class = y_train[class_mask]
            if len(y_class) == 0:
                continue

            target = args.target_per_class
            if len(y_class) >= target:
                idx = rng.choice(len(y_class), size=target, replace=False)
                x_sel = x_class[idx]
                y_sel = y_class[idx]
            else:
                x_new = [x_class]
                needed = target - len(y_class)
                for _ in range(needed):
                    src_idx = int(rng.integers(0, len(y_class)))
                    aug_seq = augment_sequence(x_class[src_idx], args.landmarks, rng)
                    x_new.append(aug_seq[np.newaxis, ...])
                x_sel = np.concatenate(x_new, axis=0)
                y_sel = np.full((target,), label, dtype=y_train.dtype)

            x_parts.append(x_sel)
            y_parts.append(y_sel)

        X_train_aug = np.concatenate(x_parts, axis=0)
        y_train_aug = np.concatenate(y_parts, axis=0)
    else:
        x_parts = [X_train]
        y_parts = [y_train]

        for _ in range(max(0, args.aug_per_sample)):
            batch = np.stack(
                [augment_sequence(seq, args.landmarks, rng) for seq in X_train],
                axis=0,
            )
            x_parts.append(batch)
            y_parts.append(y_train.copy())

        X_train_aug = np.concatenate(x_parts, axis=0)
        y_train_aug = np.concatenate(y_parts, axis=0)

    shuffle_idx = rng.permutation(len(y_train_aug))
    X_train_aug = X_train_aug[shuffle_idx]
    y_train_aug = y_train_aug[shuffle_idx]

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    flat = X_train_aug.reshape(X_train_aug.shape[0], -1)
    out_df = pd.DataFrame(flat, columns=feature_cols)
    out_df.insert(0, "label", y_train_aug)
    out_df.to_csv(args.out, index=False)

    print(f"Saved: {args.out}")
    print(f"train original={len(y_train)} train after aug={len(y_train_aug)}")
    print("Label counts after augmentation:")
    print(out_df["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
