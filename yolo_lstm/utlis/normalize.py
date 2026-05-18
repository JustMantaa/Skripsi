import numpy as np


def normalize_keypoint_sequence(sequence):
    t, f = sequence.shape
    seq3 = sequence.reshape(t, 21, 3).astype(np.float32)
    out = np.empty_like(seq3)
    wrist_ref = seq3[0, 0:1, :]

    for i in range(t):
        frame = seq3[i]
        centered = frame - wrist_ref
        scale = float(np.linalg.norm(frame[9] - frame[0]))
        if scale < 1e-6:
            scale = 1.0
        out[i] = centered / scale

    return out.reshape(t, f)
