"""
Fraud Detection Model Training Script

This script handles the training of neural network models for fraud detection.
"""

from models.fraud_models import get_model
from data.dataloader import FraudDataLoader, create_data_loaders
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudTrainer:
    """Trainer class for fraud detection models"""

    def __init__(self,
                 model_type: str = 'nn',
                 model_params: dict = None,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5,
                 device: str = None):
        """
        Initialize the trainer

        Args:
            model_type: Type of model to train
            model_params: Model-specific parameters
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            device: Device to train on ('cuda' or 'cpu')
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Set device
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_history = {'loss': [], 'accuracy': [],
                              'val_loss': [], 'val_accuracy': []}

    def prepare_data(self, data_path: str = "dataset", use_synthetic: bool = True):
        """Prepare the training data"""
        loader = FraudDataLoader(data_path)

        if use_synthetic:
            logger.info("Creating synthetic fraud detection data...")
            df = loader.create_synthetic_data(num_samples=50000)
        else:
            logger.info("Loading real fraud detection data...")
            df = loader.load_sample_data(num_files=10)
            if df is None:
                logger.warning(
                    "Failed to load real data, falling back to synthetic data")
                df = loader.create_synthetic_data(num_samples=50000)

        # Get data splits
        X_train, X_val, X_test, y_train, y_val, y_test = loader.get_data_splits(
            df)

        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            X_train, X_val, X_test, y_train, y_val, y_test, batch_size=256
        )

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.input_size = X_train.shape[1]
        self.feature_columns = loader.feature_columns

        logger.info(
            f"Data prepared: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val, {len(test_loader.dataset)} test")
        logger.info(f"Input features: {self.input_size}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def build_model(self):
        """Build the model"""
        self.model = get_model(
            self.model_type, self.input_size, **self.model_params)
        self.model = self.model.to(self.device)

        # Setup optimizer and loss function
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Use weighted loss for imbalanced data
        if hasattr(self, 'train_loader'):
            # Calculate class weights
            y_train = []
            for _, labels in self.train_loader:
                y_train.extend(labels.numpy())
            y_train = np.array(y_train)

            class_counts = np.bincount(y_train)
            total_samples = len(y_train)
            class_weights = total_samples / (len(class_counts) * class_counts)
            class_weights = torch.FloatTensor(class_weights).to(self.device)

            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            logger.info(
                f"Using weighted loss with class weights: {class_weights}")
        else:
            self.criterion = nn.CrossEntropyLoss()

        logger.info(f"Model built: {self.model.__class__.__name__}")
        logger.info(
            f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                total_loss += self.criterion(output, target).item()

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def train(self, epochs: int = 100, patience: int = 10, save_path: str = "outputs"):
        """
        Train the model with early stopping

        Args:
            epochs: Maximum number of epochs
            patience: Early stopping patience
            save_path: Path to save the model
        """
        os.makedirs(save_path, exist_ok=True)

        best_val_loss = float('inf')
        patience_counter = 0

        logger.info(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            # Update history
            self.train_history['loss'].append(train_loss)
            self.train_history['accuracy'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_accuracy'].append(val_acc)

            logger.info(f"Epoch {epoch+1}/{epochs}:")
            logger.info(
                f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Save best model
                model_path = os.path.join(
                    save_path, f"best_model_{self.model_type}.pth")
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'model_type': self.model_type,
                    'model_params': self.model_params,
                    'input_size': self.input_size,
                    'feature_columns': self.feature_columns
                }, model_path)

                logger.info(f"  ✓ New best model saved to {model_path}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Save training history
        history_path = os.path.join(
            save_path, f"training_history_{self.model_type}.json")
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)

        logger.info("Training completed!")
        return model_path

    def evaluate(self, model_path: str = None, save_path: str = "outputs"):
        """
        Evaluate the model on test data

        Args:
            model_path: Path to saved model (if None, uses current model)
            save_path: Path to save evaluation results
        """
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from {model_path}")

        self.model.eval()

        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="Evaluating"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                probabilities = torch.softmax(output, dim=1)
                predictions = output.argmax(dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)

        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(
            all_targets, all_predictions, average='weighted')
        recall = recall_score(all_targets, all_predictions, average='weighted')
        f1 = f1_score(all_targets, all_predictions, average='weighted')

        # ROC AUC for fraud class (class 1)
        roc_auc = roc_auc_score(all_targets, all_probabilities[:, 1])

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }

        logger.info("Test Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  ROC AUC: {roc_auc:.4f}")

        # Classification report
        report = classification_report(all_targets, all_predictions,
                                       target_names=['Normal', 'Fraud'])
        logger.info(f"\\nClassification Report:\\n{report}")

        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)

        # Save results
        os.makedirs(save_path, exist_ok=True)

        # Save metrics
        metrics_path = os.path.join(
            save_path, f"evaluation_metrics_{self.model_type}.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Save detailed report
        report_path = os.path.join(
            save_path, f"classification_report_{self.model_type}.txt")
        with open(report_path, 'w') as f:
            f.write(f"Model: {self.model_type}\\n")
            f.write(f"Evaluation Date: {datetime.now()}\\n\\n")
            f.write(f"Test Results:\\n")
            f.write(f"Accuracy: {accuracy:.4f}\\n")
            f.write(f"Precision: {precision:.4f}\\n")
            f.write(f"Recall: {recall:.4f}\\n")
            f.write(f"F1 Score: {f1:.4f}\\n")
            f.write(f"ROC AUC: {roc_auc:.4f}\\n\\n")
            f.write(f"Classification Report:\\n{report}\\n")
            f.write(f"\\nConfusion Matrix:\\n{cm}")

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Fraud'],
                    yticklabels=['Normal', 'Fraud'])
        plt.title(f'Confusion Matrix - {self.model_type.upper()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        cm_path = os.path.join(
            save_path, f"confusion_matrix_{self.model_type}.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Plot training history
        if self.train_history['loss']:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # Loss plot
            ax1.plot(self.train_history['loss'], label='Train Loss')
            ax1.plot(self.train_history['val_loss'], label='Validation Loss')
            ax1.set_title('Model Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)

            # Accuracy plot
            ax2.plot(self.train_history['accuracy'], label='Train Accuracy')
            ax2.plot(self.train_history['val_accuracy'],
                     label='Validation Accuracy')
            ax2.set_title('Model Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()
            history_plot_path = os.path.join(
                save_path, f"training_history_{self.model_type}.png")
            plt.savefig(history_plot_path, dpi=300, bbox_inches='tight')
            plt.close()

        logger.info(f"Evaluation results saved to {save_path}")
        return metrics


def main():
    """Main training function"""
    # Configuration
    config = {
        'model_type': 'nn',  # 'nn', 'lstm', 'autoencoder', 'ensemble'
        'model_params': {
            'hidden_sizes': [512, 256, 128, 64],
            'dropout_rate': 0.3
        },
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'epochs': 50,
        'patience': 10,
        'use_synthetic': True,
        'save_path': 'outputs'
    }

    logger.info("Starting Fraud Detection Model Training")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")

    # Initialize trainer
    trainer = FraudTrainer(
        model_type=config['model_type'],
        model_params=config['model_params'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Prepare data
    trainer.prepare_data(use_synthetic=config['use_synthetic'])

    # Build model
    trainer.build_model()

    # Train model
    model_path = trainer.train(
        epochs=config['epochs'],
        patience=config['patience'],
        save_path=config['save_path']
    )

    # Evaluate model
    trainer.evaluate(model_path, save_path=config['save_path'])

    logger.info("Training and evaluation completed successfully!")


if __name__ == "__main__":
    main()
