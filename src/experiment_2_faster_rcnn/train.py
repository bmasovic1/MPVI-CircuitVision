"""Trening Faster R-CNN-a na DeepPCB trening skupu.

Primjeri (iz korijena repoa):
    # probni trening na 16 slika, da se vidi da sve radi i da loss pada
    python -m src.experiment_2_faster_rcnn.train --limit 16 --epochs 5 --out runs/exp2_probni

    # pravi trening, checkpointi na Google Drive (Colab)
    python -m src.experiment_2_faster_rcnn.train --epochs 24 --out /content/drive/MyDrive/circuitvision/exp2/run1

    # nastavak prekinutog treninga (npr. kad Colab prekine sesiju)
    python -m src.experiment_2_faster_rcnn.train --epochs 24 --out <isti folder> --resume <isti folder>/last.pth

U --out se snimaju:
    config.json      postavke treninga
    log.csv          loss po epohi
    last.pth         zadnji checkpoint (model + optimizator), za nastavak treninga
    model_final.pth  samo težine modela nakon zadnje epohe (za predikcije)
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.experiment_2_faster_rcnn.dataset import DeepPCBDetection, RandomFlip, collate_fn
from src.experiment_2_faster_rcnn.model import build_model

LOSS_KEYS = ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]


def parse_args():
    p = argparse.ArgumentParser(description="Trening Faster R-CNN na DeepPCB-u")
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.01, help="početni learning rate (SGD)")
    p.add_argument("--min-size", type=int, default=800, help="kraća strana slike unutar modela")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="koristi samo prvih N slika (0 = sve); za probni trening")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", default="runs/exp2")
    p.add_argument("--resume", default="", help="putanja do last.pth za nastavak treninga")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_lr_lambda(iters_per_epoch, epochs, warmup_iters=200):
    """Learning rate kao faktor početnog: linearno zagrijavanje na početku, zatim pad x0.1
    na 2/3 i 8/9 treninga (uobičajen raspored za Faster R-CNN)."""
    warmup_iters = min(warmup_iters, iters_per_epoch)
    milestones = sorted({m for m in (round(epochs * 2 / 3), round(epochs * 8 / 9)) if 0 < m < epochs})

    def lr_lambda(it):
        warm = min(1.0, (it + 1) / warmup_iters)
        epoch = it // iters_per_epoch
        decay = 0.1 ** sum(epoch >= m for m in milestones)
        return warm * decay

    return lr_lambda


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dataset = DeepPCBDetection("train", transforms=RandomFlip())
    if args.limit:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        collate_fn=collate_fn, drop_last=len(dataset) > args.batch_size)

    model = build_model(min_size=args.min_size).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(len(loader), args.epochs))
    use_amp = device.type == "cuda"  # mješovita preciznost: brži trening na T4 GPU-u
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Nastavljam od epohe {start_epoch + 1}")

    config = {**vars(args), "device": str(device), "train_images": len(dataset),
              "iters_per_epoch": len(loader), "gpu": torch.cuda.get_device_name(0) if use_amp else None}
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2))

    log_path = out / "log.csv"
    if start_epoch == 0:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "loss", *LOSS_KEYS, "lr", "seconds"])

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        sums = {k: 0.0 for k in ["loss", *LOSS_KEYS]}
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets]

            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())

            if not torch.isfinite(loss):
                raise RuntimeError(f"loss nije konačan broj ({loss.item()}); smanji --lr")

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            sums["loss"] += loss.item()
            for k in LOSS_KEYS:
                sums[k] += loss_dict[k].item()

        n = len(loader)
        means = {k: v / n for k, v in sums.items()}
        lr = optimizer.param_groups[0]["lr"]
        secs = time.time() - t0
        print(f"epoha {epoch + 1}/{args.epochs}  loss {means['loss']:.4f}  "
              f"(cls {means['loss_classifier']:.3f}, box {means['loss_box_reg']:.3f}, "
              f"rpn {means['loss_objectness']:.3f}, rpn_box {means['loss_rpn_box_reg']:.3f})  "
              f"lr {lr:.5f}  {secs:.0f}s")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, *(round(means[k], 5) for k in ["loss", *LOSS_KEYS]),
                                    lr, round(secs, 1)])

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "args": vars(args)}, out / "last.pth")

    torch.save(model.state_dict(), out / "model_final.pth")
    print(f"Gotovo. Model snimljen u {out / 'model_final.pth'}")


if __name__ == "__main__":
    main()
