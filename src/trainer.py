"""
Training loop for the Beetles Drought Prediction project.

This module implements the training and validation logic, including:
- Training loop with progress bars
- Validation evaluation
- TensorBoard logging
- Model checkpointing
- Early stopping

For beginners: Training a neural network involves repeatedly:
1. Showing it examples (forward pass)
2. Measuring how wrong it is (compute loss)
3. Adjusting its weights to be less wrong (backward pass + optimizer step)
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, CHECKPOINTS_DIR, LOGS_DIR, get_device,
    LR_TYPE
)
from src.loss import GaussianNLLLoss
from src.metrics import compute_crps_batch, compute_crps_per_target, compute_competition_score
from src.utils import (
    AverageMeter, EarlyStopping, save_checkpoint,
    format_time, ensure_dir, get_timestamp
)


class Trainer:
    """
    Handles the training and validation of the regression model.

    This class encapsulates the entire training process:
    - Setting up optimizer and loss function
    - Running training epochs
    - Evaluating on validation data
    - Logging metrics to TensorBoard
    - Saving checkpoints
    - Early stopping

    Example:
        trainer = Trainer(model, train_loader, val_loader)
        trainer.train(num_epochs=100)
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        device=None,
        experiment_name=None,
        lr_type=LR_TYPE,
        num_epochs=NUM_EPOCHS
    ):
        """
        Initialize the trainer.

        Args:
            model: The PyTorch model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization strength
            device: Device to train on (default: auto-detect)
            experiment_name: Name for this experiment (for logging)
            lr_type: Learning rate scheduler type ('cosine' or 'plateau')
            num_epochs: Number of epochs (used for cosine scheduler)
        """
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr_type = lr_type

        # Set up the loss function
        # Gaussian NLL loss trains both mean predictions AND uncertainty
        self.criterion = GaussianNLLLoss()

        # Set up the optimizer
        # Adam is a popular choice that adapts learning rates automatically
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Set up learning rate scheduler
        if lr_type == 'cosine':
            # CosineAnnealingLR smoothly decreases LR following a cosine curve
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs,  # Full cosine cycle over training
                eta_min=1e-6       # Minimum LR
            )
        else:  # 'plateau'
            # ReduceLROnPlateau reduces LR when validation loss plateaus
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,   # Reduce LR by half
                patience=10   # Wait 10 epochs before reducing
            )

        # Set up experiment logging
        self.experiment_name = experiment_name or f"experiment_{get_timestamp()}"
        self.log_dir = ensure_dir(LOGS_DIR / self.experiment_name)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # Set up checkpoint directory
        self.checkpoint_dir = ensure_dir(CHECKPOINTS_DIR)

        # Track best model
        self.best_val_crps = float('inf')
        self.best_epoch = 0

        print(f"\nTrainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  LR scheduler: {lr_type}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  TensorBoard logs: {self.log_dir}")

    def train_epoch(self, epoch):
        """
        Run one training epoch.

        An epoch is one complete pass through the training data.

        Args:
            epoch: Current epoch number

        Returns:
            float: Average training loss for this epoch
        """
        # Set model to training mode (enables dropout, etc.)
        self.model.train()

        # Track average loss
        loss_meter = AverageMeter()

        # Progress bar for this epoch
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

        for embeddings, targets in pbar:
            # Move data to device (GPU if available)
            embeddings = embeddings.to(self.device)
            targets = targets.to(self.device)

            # Zero the gradients from previous step
            # This is required because PyTorch accumulates gradients
            self.optimizer.zero_grad()

            # Forward pass: get predictions from model
            mu, sigma = self.model(embeddings)

            # Compute loss
            loss = self.criterion(mu, sigma, targets)

            # Backward pass: compute gradients
            loss.backward()

            # Update weights using the gradients
            self.optimizer.step()

            # Track loss
            loss_meter.update(loss.item(), embeddings.size(0))

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})

        return loss_meter.avg

    @torch.no_grad()
    def validate(self, epoch):
        """
        Run validation.

        Evaluates the model on the validation set without updating weights.

        Args:
            epoch: Current epoch number

        Returns:
            tuple: (val_loss, val_crps, per_target_crps)
        """
        # Set model to evaluation mode (disables dropout, etc.)
        self.model.eval()

        loss_meter = AverageMeter()
        all_mu = []
        all_sigma = []
        all_targets = []

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", leave=False)

        for embeddings, targets in pbar:
            embeddings = embeddings.to(self.device)
            targets = targets.to(self.device)

            # Forward pass only (no gradients needed)
            mu, sigma = self.model(embeddings)

            # Compute loss
            loss = self.criterion(mu, sigma, targets)
            loss_meter.update(loss.item(), embeddings.size(0))

            # Collect predictions for CRPS calculation
            all_mu.append(mu.cpu())
            all_sigma.append(sigma.cpu())
            all_targets.append(targets.cpu())

        # Concatenate all batches
        all_mu = torch.cat(all_mu, dim=0).numpy()
        all_sigma = torch.cat(all_sigma, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()

        # Compute CRPS (the competition metric)
        rms_crps, per_target_crps = compute_competition_score(
            all_mu, all_sigma, all_targets
        )

        return loss_meter.avg, rms_crps, per_target_crps

    def train(self, num_epochs=NUM_EPOCHS, early_stopping_patience=EARLY_STOPPING_PATIENCE):
        """
        Run the full training loop.

        Args:
            num_epochs: Maximum number of epochs to train
            early_stopping_patience: Stop if validation doesn't improve for this many epochs

        Returns:
            dict: Training history
        """
        print(f"\n{'='*60}")
        print("STARTING TRAINING")
        print(f"{'='*60}")
        print(f"Max epochs: {num_epochs}")
        print(f"Early stopping patience: {early_stopping_patience}")
        print(f"{'='*60}\n")

        # Initialize early stopping
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            mode='min'  # Lower CRPS is better
        )

        # Track training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_crps': [],
            'learning_rate': []
        }

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # Training
            train_loss = self.train_epoch(epoch)

            # Validation
            val_loss, val_crps, per_target_crps = self.validate(epoch)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update learning rate scheduler
            if self.lr_type == 'cosine':
                self.scheduler.step()  # Cosine scheduler steps by epoch
            else:
                self.scheduler.step(val_crps)  # Plateau scheduler needs metric

            # Track history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_crps'].append(val_crps)
            history['learning_rate'].append(current_lr)

            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('CRPS/val', val_crps, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)

            for target_name, crps_value in per_target_crps.items():
                self.writer.add_scalar(f'CRPS/{target_name}', crps_value, epoch)

            # Calculate epoch time
            epoch_time = time.time() - epoch_start

            # Print epoch summary
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val CRPS: {val_crps:.4f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {epoch_time:.1f}s")

            # Save best model
            if val_crps < self.best_val_crps:
                self.best_val_crps = val_crps
                self.best_epoch = epoch
                self.save_best_model()
                print(f"  → New best model! CRPS: {val_crps:.4f}")

            # Check early stopping
            if early_stopping(val_crps):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                print(f"Best model was at epoch {self.best_epoch} with CRPS: {self.best_val_crps:.4f}")
                break

        # Training complete
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Total time: {format_time(total_time)}")
        print(f"Best epoch: {self.best_epoch}")
        print(f"Best validation CRPS: {self.best_val_crps:.4f}")
        print(f"{'='*60}\n")

        # Close TensorBoard writer
        self.writer.close()

        return history

    def save_best_model(self):
        """Save the current model as the best model."""
        save_path = self.checkpoint_dir / "best_model.pt"
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=self.best_epoch,
            loss=self.best_val_crps,
            path=save_path,
            extra_info={'val_crps': self.best_val_crps}
        )

    def save_final_model(self):
        """Save the final model at the end of training."""
        save_path = self.checkpoint_dir / "final_model.pt"
        torch.save(self.model.state_dict(), save_path)
        print(f"Final model saved to {save_path}")

    def load_best_model(self):
        """Load the best saved model."""
        load_path = self.checkpoint_dir / "best_model.pt"
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")
        return self.model


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    experiment_name=None,
    device=None
):
    """
    Convenience function to train a model.

    This is a simpler interface if you don't need fine-grained control.

    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs
        learning_rate: Learning rate
        experiment_name: Name for the experiment
        device: Device to train on

    Returns:
        tuple: (trained_model, history)
    """
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        experiment_name=experiment_name,
        device=device
    )

    history = trainer.train(num_epochs=num_epochs)

    # Load the best model
    best_model = trainer.load_best_model()

    return best_model, history


if __name__ == "__main__":
    # Quick test with dummy data
    print("Testing Trainer with dummy data...")

    from torch.utils.data import TensorDataset, DataLoader
    from src.model import RegressionHead

    # Create dummy data
    num_train = 200
    num_val = 50
    embedding_dim = 768
    num_targets = 3

    train_embeddings = torch.randn(num_train, embedding_dim)
    train_targets = torch.randn(num_train, num_targets)
    val_embeddings = torch.randn(num_val, embedding_dim)
    val_targets = torch.randn(num_val, num_targets)

    train_dataset = TensorDataset(train_embeddings, train_targets)
    val_dataset = TensorDataset(val_embeddings, val_targets)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create model
    model = RegressionHead(embedding_dim=embedding_dim)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_name="test_run"
    )

    # Train for a few epochs
    print("\nRunning short training test...")
    history = trainer.train(num_epochs=3, early_stopping_patience=5)

    print("\nTest complete!")
    print(f"Training losses: {history['train_loss']}")
    print(f"Validation CRPS: {history['val_crps']}")
