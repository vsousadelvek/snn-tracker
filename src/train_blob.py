"""Treino do classificador de blobs. Uso: python -m src.train_blob --epochs 30"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from .blob_snn import BlobSNN, CLASSES


class BlobDataset(Dataset):
    def __init__(self, root="data_blobs"):
        self.samples = []
        for i, c in enumerate(CLASSES):
            for f in sorted((Path(root) / c).glob("*.npy")):
                self.samples.append((f, i))
        if not self.samples:
            raise SystemExit("sem dados — rode src.harvest_blobs primeiro")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        f, y = self.samples[i]
        x = torch.from_numpy(np.load(f).astype(np.float32))
        if torch.rand(1) < 0.5:           # augment: flip horizontal
            x = torch.flip(x, dims=[2])
        return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = BlobDataset()
    n_test = max(1, int(0.2 * len(ds)))
    tr, te = random_split(ds, [len(ds) - n_test, n_test],
                          generator=torch.Generator().manual_seed(0))
    print(f"device={device} train={len(tr)} test={len(te)}")

    labels = [ds.samples[i][1] for i in tr.indices]
    counts = np.bincount(labels, minlength=len(CLASSES))
    weights = torch.tensor([1.0 / counts[l] for l in labels])
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

    dl = DataLoader(tr, batch_size=args.batch_size, sampler=sampler)
    dl_te = DataLoader(te, batch_size=64)
    model = BlobSNN(len(CLASSES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in dl_te:
                correct += (model(x.to(device)).argmax(1) == y.to(device)).sum().item()
                total += y.size(0)
        acc = correct / total
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "classes": CLASSES},
                       "checkpoints/blob_snn.pt")
        if ep % 5 == 0 or ep == 1:
            print(f"epoch {ep}: test_acc={acc:.3f} (best={best:.3f})")
    print(f"melhor acc={best:.3f} -> checkpoints/blob_snn.pt")


if __name__ == "__main__":
    main()
