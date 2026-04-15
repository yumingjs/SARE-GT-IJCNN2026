"""Training script for SARE-GT on the corporate credit-rating task.

Usage::

    python train.py [--epochs 600] [--hidden_dim 256] [--lr 0.002] ...

See ``python train.py --help`` for the full list of arguments.
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import trange

from model import SARE_GT
from dataset import CCRPeerNetworkDataset
from utils import (
    set_seed, setup_logger, WarmupCosineScheduler, compute_metrics,
)


def train(model, data, args, logger):
    """End-to-end SARE-GT training with warmup + cosine schedule.

    Args:
        model: ``SARE_GT`` instance (already on device).
        data: PyG ``Data`` object (already on device).
        args: Parsed CLI arguments.
        logger: Python logger.

    Returns:
        dict: Best evaluation metrics observed during training.
    """
    base_lr = min(args.lr * 0.8, 0.0015)
    warmup = args.warmup_epochs * 2

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=warmup, total_epochs=args.epochs,
        base_lr=base_lr, min_lr=args.min_lr)

    logger.info(f"Optimizer: AdamW (lr={base_lr}, wd={args.weight_decay})")
    logger.info(f"Schedule: Warmup({warmup}) + Cosine({args.epochs})")

    best_metrics = {"f1_macro": 0.0}
    best_epoch = 0

    for epoch in trange(args.epochs, desc="SARE-GT Training"):
        lr = scheduler.step(epoch)

        # -- train step --
        model.train()
        optimizer.zero_grad()
        logits, _ = model(data)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # -- evaluation --
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                logits, _ = model(data)
                preds = logits[data.test_mask].argmax(dim=1).cpu().numpy()
                labels = data.y[data.test_mask].cpu().numpy()
                metrics = compute_metrics(labels, preds)

            logger.info(
                f"Epoch {epoch+1}/{args.epochs} | LR {lr:.6f} | "
                f"Loss {loss.item():.5f} | "
                f"Acc {metrics['accuracy']:.4f} | "
                f"F1 {metrics['f1_macro']:.4f} | "
                f"QWK {metrics['qwk']:.4f}"
            )

            if metrics["f1_macro"] > best_metrics["f1_macro"]:
                best_metrics = metrics
                best_epoch = epoch

    logger.info("=" * 60)
    logger.info(f"Best results @ epoch {best_epoch + 1}:")
    for k, v in best_metrics.items():
        logger.info(f"  {k}: {v:.6f}")
    logger.info("=" * 60)

    return best_metrics


def main():
    parser = argparse.ArgumentParser(
        description="SARE-GT training for corporate credit rating")
    parser.add_argument("--data_root", type=str, default="./CCRDataset",
                        help="Root directory of the dataset")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--warmup_epochs", type=int, default=100)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k_neighbors", type=int, default=15)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Number of GAT layers in Local Expert")
    parser.add_argument("--transformer_layers", type=int, default=2,
                        help="Number of Graph Transformer layers")
    parser.add_argument("--pe_dim", type=int, default=16,
                        help="Laplacian positional encoding dimension")
    parser.add_argument("--eval_freq", type=int, default=10)
    parser.add_argument("--save_path", type=str,
                        default="sare_gt_model.pth")
    args = parser.parse_args()

    set_seed(args.seed)
    logger, log_file = setup_logger("sare_gt")

    logger.info("SARE-GT: Structure-Aware Robust Enhancement Graph Transformer")
    logger.info(f"Configuration: {vars(args)}")

    # Load dataset
    dataset = CCRPeerNetworkDataset(
        root=args.data_root,
        k_neighbors=args.k_neighbors,
        use_pe=True,
        pe_dim=args.pe_dim,
        use_topo=True,
        topo_features=["degree", "pagerank", "hub_score"],
    )
    data = dataset[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    topo_dim = getattr(data, "topo_dim", 0)
    num_classes = int(data.y.max().item()) + 1

    logger.info(f"Nodes: {data.x.size(0)} | Edges: {data.edge_index.size(1)} | "
                f"Features: {data.x.size(1)} | Classes: {num_classes}")

    # Build model
    model = SARE_GT(
        input_dim=data.x.size(1),
        hidden_dim=args.hidden_dim,
        output_dim=num_classes,
        num_layers=args.num_layers,
        topo_dim=topo_dim,
        pe_dim=args.pe_dim,
        transformer_layers=args.transformer_layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params:,}")

    # Train
    best = train(model, data, args, logger)

    # Save
    torch.save(model.state_dict(), args.save_path)
    logger.info(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
