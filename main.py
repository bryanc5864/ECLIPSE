#!/usr/bin/env python3
"""
ECLIPSE: Main Entry Point

Command-line interface for ECLIPSE framework.
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_data(args):
    """Download public datasets."""
    from src.data import DataDownloader

    logger.info("Starting data download...")
    downloader = DataDownloader(args.data_dir)
    results = downloader.download_all(skip_large=args.skip_large)

    for source, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"  {source}: {status}")


def train_model(args):
    """Train a model."""
    import torch
    from pathlib import Path
    from src.data import (
        AmpliconRepositoryLoader, CytoCellDBLoader, DepMapLoader, HiCLoader,
        ECDNADataset, DynamicsDataset, VulnerabilityDataset, create_dataloader,
        SplitGenerator
    )
    from src.models import ECDNAFormer, CircularODE, VulnCausal, ECLIPSE
    from src.training import (
        ECDNAFormerTrainer, CircularODETrainer, VulnCausalTrainer, ECLIPSETrainer
    )

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    logger.info(f"Training on device: {device}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Checkpoint directory: {args.checkpoint_dir}")

    data_dir = Path(args.data_dir)

    if args.module == "former":
        logger.info("Training ecDNA-Former (Module 1)...")
        logger.info("Loading ecDNA labels and features...")

        # Load ecDNA dataset
        dataset = ECDNADataset.from_data_dir(
            data_dir=data_dir,
            split="train",
        )
        logger.info(f"Training samples: {len(dataset)}")

        # Create validation set
        val_dataset = ECDNADataset.from_data_dir(
            data_dir=data_dir,
            split="val",
        )
        logger.info(f"Validation samples: {len(val_dataset)}")

        # Create dataloaders
        train_loader = create_dataloader(
            dataset, batch_size=args.batch_size, shuffle=True
        )
        val_loader = create_dataloader(
            val_dataset, batch_size=args.batch_size, shuffle=False
        )

        # Create model
        model = ECDNAFormer()
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Create trainer
        trainer = ECDNAFormerTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            use_wandb=args.wandb,
        )

        # Train
        history = trainer.train(num_epochs=args.epochs, early_stopping_patience=args.patience)
        logger.info("ecDNA-Former training complete")
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")

    elif args.module == "dynamics":
        logger.info("Training CircularODE (Module 2)...")

        # Load trajectory dataset
        dataset = DynamicsDataset.from_data_dir(
            data_dir=data_dir / "ecdna_trajectories",
            split="train",
        )
        logger.info(f"Training trajectories: {len(dataset)}")

        val_dataset = DynamicsDataset.from_data_dir(
            data_dir=data_dir / "ecdna_trajectories",
            split="val",
        )

        train_loader = create_dataloader(
            dataset, batch_size=args.batch_size, shuffle=True
        )
        val_loader = create_dataloader(
            val_dataset, batch_size=args.batch_size, shuffle=False
        )

        model = CircularODE()
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        trainer = CircularODETrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            use_wandb=args.wandb,
        )

        history = trainer.train(num_epochs=args.epochs)
        logger.info("CircularODE training complete")

    elif args.module == "vuln":
        logger.info("Training VulnCausal (Module 3)...")

        # Load vulnerability dataset
        dataset = VulnerabilityDataset.from_data_dir(
            data_dir=data_dir,
            split="train",
        )
        logger.info(f"Training cell lines: {len(dataset)}")

        val_dataset = VulnerabilityDataset.from_data_dir(
            data_dir=data_dir,
            split="val",
        )

        train_loader = create_dataloader(
            dataset, batch_size=args.batch_size, shuffle=True
        )
        val_loader = create_dataloader(
            val_dataset, batch_size=args.batch_size, shuffle=False
        )

        # Get data dimensions from dataset
        num_genes = dataset.crispr.shape[1]
        expression_dim = dataset.expression.shape[1]
        logger.info(f"Data dimensions: {num_genes} genes, {expression_dim} expression features")

        model = VulnCausal(
            num_genes=num_genes,
            expression_dim=expression_dim,
        )
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        trainer = VulnCausalTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            use_wandb=args.wandb,
        )

        history = trainer.train(num_epochs=args.epochs)
        logger.info("VulnCausal training complete")

    elif args.module == "eclipse":
        logger.info("Training full ECLIPSE...")
        model = ECLIPSE()
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        # Full ECLIPSE training requires all data sources
        logger.warning("Full ECLIPSE training not yet implemented - train modules separately first")
        logger.info("ECLIPSE training complete")


def predict(args):
    """Run prediction with trained model."""
    import torch
    from src.models import ECLIPSE

    logger.info(f"Loading model from {args.checkpoint}...")
    model = ECLIPSE.from_pretrained(args.checkpoint)
    model.eval()

    # Load input data
    logger.info(f"Loading input from {args.input}...")
    # (Placeholder for actual prediction)

    logger.info("Prediction complete")


def evaluate(args):
    """Evaluate model on validation set."""
    import torch
    from src.utils.metrics import compute_all_metrics

    logger.info(f"Evaluating model {args.checkpoint}...")
    # (Placeholder for actual evaluation)

    logger.info("Evaluation complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ECLIPSE: ecDNA prediction and vulnerability discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download data
  python main.py download --data-dir data

  # Train ecDNA-Former
  python main.py train --module former --epochs 100

  # Run prediction
  python main.py predict --checkpoint checkpoints/eclipse.pt --input sample.pt

  # Evaluate model
  python main.py evaluate --checkpoint checkpoints/eclipse.pt --val-data val.pt
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download datasets")
    download_parser.add_argument("--data-dir", type=str, default="data",
                                 help="Directory to save data")
    download_parser.add_argument("--skip-large", action="store_true",
                                 help="Skip large files (Hi-C)")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--module", type=str, required=True,
                              choices=["former", "dynamics", "vuln", "eclipse"],
                              help="Module to train")
    train_parser.add_argument("--data-dir", type=str, default="data",
                              help="Data directory")
    train_parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                              help="Checkpoint directory")
    train_parser.add_argument("--epochs", type=int, default=100,
                              help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=32,
                              help="Batch size")
    train_parser.add_argument("--patience", type=int, default=5,
                              help="Early stopping patience")
    train_parser.add_argument("--lr", type=float, default=1e-4,
                              help="Learning rate")
    train_parser.add_argument("--cpu", action="store_true",
                              help="Use CPU only")
    train_parser.add_argument("--wandb", action="store_true",
                              help="Use Weights & Biases logging")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run prediction")
    predict_parser.add_argument("--checkpoint", type=str, required=True,
                                help="Model checkpoint path")
    predict_parser.add_argument("--input", type=str, required=True,
                                help="Input data path")
    predict_parser.add_argument("--output", type=str, default="predictions.pt",
                                help="Output path")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model")
    eval_parser.add_argument("--checkpoint", type=str, required=True,
                             help="Model checkpoint path")
    eval_parser.add_argument("--val-data", type=str, required=True,
                             help="Validation data path")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "download":
        download_data(args)
    elif args.command == "train":
        train_model(args)
    elif args.command == "predict":
        predict(args)
    elif args.command == "evaluate":
        evaluate(args)


if __name__ == "__main__":
    main()
