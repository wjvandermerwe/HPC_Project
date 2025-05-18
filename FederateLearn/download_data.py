#!/usr/bin/env python3
import os, struct, numpy as np
from tensorflow.keras.datasets import cifar10

# ------------------------------------------------------------------#
# Config                                                             #
# ------------------------------------------------------------------#
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

clients = {
    "A": ([0, 1, 2], 5000, 1000, 0.7),
    "B": ([3, 4, 5], 3000,  500, 0.7),
    "C": ([6, 7],    2000,  400, 0.7),
    "D": ([8, 9],    1000,  200, 0.7),
}

(train_X, train_y), (test_X, test_y) = cifar10.load_data()
train_X, test_X = train_X.astype(np.float64)/255.0, test_X.astype(np.float64)/255.0
train_y, test_y = train_y.flatten(), test_y.flatten()

def build_skewed_shard(X, y, fav_labels, prob_fav, n_samples):
    fav_mask    = np.isin(y, fav_labels)
    other_mask  = ~fav_mask

    fav_idx     = np.where(fav_mask)[0]
    other_idx   = np.where(other_mask)[0]

    n_fav   = int(n_samples * prob_fav)
    n_other = n_samples - n_fav

    sel = np.concatenate([
        np.random.choice(fav_idx,   n_fav,   replace=False),
        np.random.choice(other_idx, n_other, replace=False)
    ])
    np.random.shuffle(sel)
    return X[sel]

def write_binary(name, X, out_dir=output_dir):
    n, H, W, C = X.shape
    D = H * W * C
    flat = X.reshape(n, D)
    path = os.path.join(out_dir, f"{name}.bin")
    with open(path, "wb") as f:
        f.write(struct.pack("ii", n, D))
        f.write(flat.tobytes())
    print(f"Wrote {path}: {n} × {D}")

for cid, (fav, n_tr, n_te, bias) in clients.items():
    Xtr = build_skewed_shard(train_X, train_y, fav, bias, n_tr)
    Xte = build_skewed_shard(test_X,  test_y,  fav, bias, n_te)

    write_binary(f"train_client_{cid}", Xtr)
    write_binary(f"test_client_{cid}",  Xte)

write_binary(
    "tests/data/test_client_A_100",
    build_skewed_shard(test_X, test_y, [0,1,2], 0.7, 100),
    out_dir="."
)

print("All shards written.")
