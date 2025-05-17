#!/usr/bin/env python3
import os
import struct
import numpy as np
from tensorflow.keras.datasets import cifar10

# Output directory
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# Load CIFAR-10
(train_X, train_y), (test_X, test_y) = cifar10.load_data()
train_X = train_X.astype(np.float64) / 255.0
test_X  = test_X.astype(np.float64)  / 255.0
train_y = train_y.flatten()
test_y  = test_y.flatten()

# Define clients: (label_subset, num_train, num_test)
clients = {
    "A":  ( [0,1,2], 5000, 1000 ),
    "B":  ( [3,4,5], 3000,  500 ),
    "C":  ( [6,7],   2000,  400 ),
    "D":  ( [8,9],   1000,  200 ),
}

def build_shard(X, y, labels, max_samples):
    # filter by label subset
    mask = np.isin(y, labels)
    Xf = X[mask]
    yf = y[mask]
    # shuffle and truncate
    perm = np.random.permutation(len(Xf))
    idx  = perm[:max_samples]
    return Xf[idx], yf[idx]

def write_binary(name, X):
    n, H, W, C = X.shape
    D = H * W * C
    flat = X.reshape(n, D)
    path = os.path.join(output_dir, f"{name}.bin")
    with open(path, "wb") as f:
        # header
        f.write(struct.pack("ii", n, D))
        # payload
        f.write(flat.tobytes())
    print(f"Wrote {path}: {n} samples × {D} dims")

# Process each client
for client_id, (labels, n_train, n_test) in clients.items():
    # build train/test shards
    Xt, yt = build_shard(train_X, train_y, labels, n_train)
    Xe, ye = build_shard(test_X,  test_y,  labels, n_test)
    # write them
    write_binary(f"train_client_{client_id}", Xt)
    write_binary(f"test_client_{client_id}",  Xe)

print("All shards written.")
