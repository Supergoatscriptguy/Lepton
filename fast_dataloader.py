"""Fast streaming dataloader for Lepton training."""

import os, random
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader


class FastShardDataset(IterableDataset):
    def __init__(self, data_dir, seq_len=512, split='train', train_ratio=0.98, shuffle=True):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.shuffle = shuffle

        shard_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith('shard_') and f.endswith('.npy')])
        split_idx = int(len(shard_files) * train_ratio)
        self.shard_files = shard_files[:split_idx] if split == 'train' else shard_files[split_idx:]
        print(f"[{split}] {len(self.shard_files)} shards")

    def __iter__(self):
        shard_order = list(range(len(self.shard_files)))
        if self.shuffle: random.shuffle(shard_order)

        while True:
            for shard_idx in shard_order:
                shard = np.load(self.shard_files[shard_idx])
                indices = list(range(len(shard)))
                if self.shuffle: random.shuffle(indices)
                for idx in indices:
                    seq = shard[idx]
                    if len(seq) >= self.seq_len:
                        yield torch.from_numpy(seq[:self.seq_len].astype(np.int64))
            if self.shuffle: random.shuffle(shard_order)


def create_fast_dataloaders(data_dir, batch_size=16, seq_len=512):
    train_ds = FastShardDataset(data_dir, seq_len=seq_len, split='train', shuffle=True)
    val_ds = FastShardDataset(data_dir, seq_len=seq_len, split='val', shuffle=False)
    return DataLoader(train_ds, batch_size=batch_size), DataLoader(val_ds, batch_size=batch_size)


if __name__ == '__main__':
    import time, argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    args = p.parse_args()

    train_loader, _ = create_fast_dataloaders(args.data_dir, batch_size=16, seq_len=512)
    start, tokens = time.time(), 0
    for i, batch in enumerate(train_loader):
        tokens += batch.numel()
        if i >= 100: break
    print(f"{tokens/(time.time()-start)/1000:.1f}k tok/s")
