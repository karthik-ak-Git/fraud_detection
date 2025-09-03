"""
Enhanced Training Script for 99% Accuracy

Advanced training pipeline with ensemble methods, cross-validation, and hyperparameter optimization.
"""

from data.enhanced_dataloader import EnhancedFraudDataLoader, create_enhanced_data_loaders
from models.enhanced_fraud_models import (
    EnhancedFraudDetectionNN,
    FraudDetectionTransformer,
    MultiScaleFraudDetectionNN,
    FraudDetectionLSTM
)
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
import json
import joblib
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced components

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class EnhancedTrainer:
    """Enhanced trainer for achieving 99% accuracy"""

    def __init__(self, device=None):
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        self.models = {}
        self.training_history = {}
        self.best_models = {}
        self.ensemble_weights = {}

    def create_models(self, input_size: int) -> Dict:
        """Create all enhanced models"""
        models = {
            'enhanced_nn': EnhancedFraudDetectionNN(input_size),
            'transformer': FraudDetectionTransformer(input_size),
            'multiscale': MultiScaleFraudDetectionNN(input_size),
            'lstm': FraudDetectionLSTM(input_size)
        }

        # Move models to device
        for name, model in models.items():
            models[name] = model.to(self.device)

        return models

    def train_single_model(self, model, train_loader, val_loader, model_name: str,
                           epochs: int = 100, lr: float = 0.001) -> Dict:
        """Train a single model with advanced techniques"""

        logger.info(f"Training {model_name}...")

        # Loss function - use Focal Loss for better class imbalance handling
        criterion = FocalLoss(alpha=2, gamma=2)

        # Optimizer with weight decay for regularization
        optimizer = optim.AdamW(model.parameters(), lr=lr,
                                weight_decay=0.01, eps=1e-8)

        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=0.7, patience=10)

        # Training history
        history = {
            'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_f1': [], 'val_auc': []
        }

        best_val_f1 = 0
        best_model_state = None
        patience_counter = 0
        max_patience = 25

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = model(data)

                # Handle different output formats
                if output.dim() == 1:
                    # Binary output
                    output = output.unsqueeze(1)
                    target_one_hot = torch.zeros(
                        target.size(0), 2).to(self.device)
                    target_one_hot.scatter_(1, target.unsqueeze(1), 1)
                    loss = criterion(
                        torch.cat([1-output, output], dim=1), target)
                else:
                    # Multi-class output
                    loss = criterion(output, target)

                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += loss.item()

            # Validation phase
            model.eval()
            val_loss = 0.0
            predictions = []
            targets = []
            probabilities = []

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)

                    # Handle different output formats
                    if output.dim() == 1:
                        # Binary output
                        prob = torch.sigmoid(output)
                        pred = (prob > 0.5).long()
                        output_for_loss = output.unsqueeze(1)
                        target_one_hot = torch.zeros(
                            target.size(0), 2).to(self.device)
                        target_one_hot.scatter_(1, target.unsqueeze(1), 1)
                        loss = criterion(
                            torch.cat([1-output_for_loss, output_for_loss], dim=1), target)
                    else:
                        # Multi-class output
                        # Probability of fraud class
                        prob = torch.softmax(output, dim=1)[:, 1]
                        pred = torch.argmax(output, dim=1)
                        loss = criterion(output, target)

                    val_loss += loss.item()
                    predictions.extend(pred.cpu().numpy())
                    targets.extend(target.cpu().numpy())
                    probabilities.extend(prob.cpu().numpy())

            # Calculate metrics
            val_accuracy = np.mean(np.array(predictions) == np.array(targets))
            val_f1 = f1_score(targets, predictions)
            val_auc = roc_auc_score(targets, probabilities)

            # Update learning rate
            scheduler.step(val_f1)

            # Store history
            history['train_loss'].append(train_loss / len(train_loader))
            history['val_loss'].append(val_loss / len(val_loader))
            history['val_accuracy'].append(val_accuracy)
            history['val_f1'].append(val_f1)
            history['val_auc'].append(val_auc)

            # Early stopping and model saving
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = model.state_dict().copy()
                patience_counter = 0

                # Save best model
                torch.save(best_model_state, f'models/{model_name}_best.pth')
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                logger.info(f'{model_name} - Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, '
                            f'Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}')

            # Early stopping
            if patience_counter >= max_patience:
                logger.info(
                    f'Early stopping for {model_name} at epoch {epoch}')
                break

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        logger.info(
            f'{model_name} training completed. Best Val F1: {best_val_f1:.4f}')

        return {
            'model': model,
            'history': history,
            'best_f1': best_val_f1,
            'final_epoch': epoch
        }

    def cross_validate_model(self, model_class, X, y, cv_folds: int = 5) -> Dict:
        """Perform cross-validation for a model"""

        logger.info(f"Cross-validating {model_class.__name__}...")

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = {'accuracy': [], 'f1': [], 'auc': []}

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"Training fold {fold + 1}/{cv_folds}")

            # Split data
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]

            # Create data loaders
            train_loader_cv, val_loader_cv, _ = create_enhanced_data_loaders(
                X_train_cv, X_val_cv, X_val_cv, y_train_cv, y_val_cv, y_val_cv,
                batch_size=512
            )

            # Create and train model
            model = model_class(X.shape[1]).to(self.device)
            result = self.train_single_model(
                model, train_loader_cv, val_loader_cv,
                f"{model_class.__name__}_fold_{fold}",
                epochs=50, lr=0.001
            )

            # Evaluate on validation set
            model.eval()
            predictions = []
            targets = []
            probabilities = []

            with torch.no_grad():
                for data, target in val_loader_cv:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)

                    if output.dim() == 1:
                        prob = torch.sigmoid(output)
                        pred = (prob > 0.5).long()
                    else:
                        prob = torch.softmax(output, dim=1)[:, 1]
                        pred = torch.argmax(output, dim=1)

                    predictions.extend(pred.cpu().numpy())
                    targets.extend(target.cpu().numpy())
                    probabilities.extend(prob.cpu().numpy())

            # Calculate metrics
            accuracy = np.mean(np.array(predictions) == np.array(targets))
            f1 = f1_score(targets, predictions)
            auc = roc_auc_score(targets, probabilities)

            cv_scores['accuracy'].append(accuracy)
            cv_scores['f1'].append(f1)
            cv_scores['auc'].append(auc)

            logger.info(
                f"Fold {fold + 1} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

        # Calculate mean and std
        cv_results = {}
        for metric, scores in cv_scores.items():
            cv_results[f'{metric}_mean'] = np.mean(scores)
            cv_results[f'{metric}_std'] = np.std(scores)

        logger.info(f"CV Results for {model_class.__name__}:")
        for metric in ['accuracy', 'f1', 'auc']:
            mean_score = cv_results[f'{metric}_mean']
            std_score = cv_results[f'{metric}_std']
            logger.info(
                f"  {metric.upper()}: {mean_score:.4f} ± {std_score:.4f}")

        return cv_results

    def train_ensemble(self, train_loader, val_loader, test_loader, input_size: int):
        """Train ensemble of models for maximum accuracy"""

        logger.info("Starting ensemble training for 99% accuracy...")

        # Create models
        self.models = self.create_models(input_size)

        # Train each model
        for model_name, model in self.models.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {model_name}")
            logger.info(f"{'='*50}")

            result = self.train_single_model(
                model, train_loader, val_loader, model_name,
                epochs=150, lr=0.001
            )

            self.training_history[model_name] = result['history']
            self.best_models[model_name] = result['model']

        # Evaluate ensemble on test set
        logger.info("\nEvaluating ensemble on test set...")
        ensemble_accuracy = self.evaluate_ensemble(test_loader)

        # Calculate ensemble weights based on validation performance
        self.calculate_ensemble_weights(val_loader)

        logger.info(
            f"\n🎯 Final Ensemble Test Accuracy: {ensemble_accuracy:.4f}")

        return ensemble_accuracy

    def calculate_ensemble_weights(self, val_loader):
        """Calculate optimal ensemble weights based on validation performance"""

        logger.info("Calculating ensemble weights...")

        # Evaluate each model on validation set
        model_performances = {}

        for model_name, model in self.best_models.items():
            model.eval()
            predictions = []
            targets = []

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)

                    if output.dim() == 1:
                        pred = (torch.sigmoid(output) > 0.5).long()
                    else:
                        pred = torch.argmax(output, dim=1)

                    predictions.extend(pred.cpu().numpy())
                    targets.extend(target.cpu().numpy())

            f1 = f1_score(targets, predictions)
            model_performances[model_name] = f1
            logger.info(f"{model_name} validation F1: {f1:.4f}")

        # Calculate weights based on F1 scores
        total_performance = sum(model_performances.values())
        self.ensemble_weights = {
            name: perf / total_performance
            for name, perf in model_performances.items()
        }

        logger.info("Ensemble weights:")
        for name, weight in self.ensemble_weights.items():
            logger.info(f"  {name}: {weight:.3f}")

    def evaluate_ensemble(self, test_loader) -> float:
        """Evaluate ensemble performance"""

        all_predictions = {name: [] for name in self.best_models.keys()}
        targets = []

        # Get predictions from all models
        for model_name, model in self.best_models.items():
            model.eval()
            model_predictions = []

            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)

                    if output.dim() == 1:
                        prob = torch.sigmoid(output)
                    else:
                        prob = torch.softmax(output, dim=1)[:, 1]

                    model_predictions.extend(prob.cpu().numpy())

                    # Store targets only once
                    if model_name == list(self.best_models.keys())[0]:
                        targets.extend(target.cpu().numpy())

            all_predictions[model_name] = model_predictions

        # Ensemble prediction (weighted average)
        ensemble_probs = np.zeros(len(targets))
        for model_name, predictions in all_predictions.items():
            weight = self.ensemble_weights.get(
                model_name, 1.0 / len(self.best_models))
            ensemble_probs += weight * np.array(predictions)

        ensemble_predictions = (ensemble_probs > 0.5).astype(int)

        # Calculate metrics
        accuracy = np.mean(ensemble_predictions == np.array(targets))
        f1 = f1_score(targets, ensemble_predictions)
        auc = roc_auc_score(targets, ensemble_probs)

        logger.info(f"Ensemble Performance:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  AUC: {auc:.4f}")

        # Print detailed classification report
        report = classification_report(targets, ensemble_predictions,
                                       target_names=['Normal', 'Fraud'])
        logger.info(f"\nClassification Report:\n{report}")

        return accuracy

    def save_training_results(self):
        """Save training results and model metadata"""

        results = {
            'timestamp': datetime.now().isoformat(),
            'device': self.device,
            'model_performances': {},
            'ensemble_weights': self.ensemble_weights,
            'training_history': {}
        }

        # Save individual model performance
        for model_name in self.models.keys():
            if model_name in self.training_history:
                history = self.training_history[model_name]
                if history['val_f1']:
                    best_f1 = max(history['val_f1'])
                    best_accuracy = max(history['val_accuracy'])
                    best_auc = max(history['val_auc'])

                    results['model_performances'][model_name] = {
                        'best_f1': best_f1,
                        'best_accuracy': best_accuracy,
                        'best_auc': best_auc
                    }

                    # Store simplified history (last 10 epochs)
                    results['training_history'][model_name] = {
                        'val_f1': history['val_f1'][-10:],
                        'val_accuracy': history['val_accuracy'][-10:],
                        'val_auc': history['val_auc'][-10:]
                    }

        # Save to file
        with open('training_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        logger.info("Training results saved to training_results.json")

    def plot_training_history(self):
        """Plot training history for all models"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Enhanced Model Training History', fontsize=16)

        metrics = ['val_accuracy', 'val_f1', 'val_auc', 'val_loss']
        titles = ['Validation Accuracy', 'Validation F1 Score',
                  'Validation AUC', 'Validation Loss']

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]

            for model_name, history in self.training_history.items():
                if metric in history and history[metric]:
                    ax.plot(history[metric], label=model_name, linewidth=2)

            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

        logger.info("Training history plot saved as training_history.png")


def main():
    """Main training function"""

    logger.info("🚀 Starting Enhanced Fraud Detection Training for 99% Accuracy")

    # Initialize data loader
    data_loader = EnhancedFraudDataLoader("dataset")

    # Create high-quality synthetic data
    logger.info("Creating high-quality synthetic dataset...")
    df = data_loader.create_high_quality_synthetic_data(num_samples=200000)

    # Get enhanced data splits
    logger.info("Preparing enhanced data splits...")
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.get_enhanced_data_splits(
        df, use_balancing=True
    )

    # Create data loaders
    train_loader, val_loader, test_loader = create_enhanced_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test,
        batch_size=512, num_workers=0
    )

    # Initialize trainer
    trainer = EnhancedTrainer()

    # Train ensemble
    final_accuracy = trainer.train_ensemble(
        train_loader, val_loader, test_loader, X_train.shape[1]
    )

    # Save results
    trainer.save_training_results()

    # Plot training history
    try:
        trainer.plot_training_history()
    except Exception as e:
        logger.warning(f"Could not create plots: {e}")

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"🎯 TRAINING COMPLETED!")
    logger.info(f"📊 Final Ensemble Accuracy: {final_accuracy:.4f}")
    logger.info(
        f"🎯 Target Accuracy (99%): {'✅ ACHIEVED' if final_accuracy >= 0.99 else '❌ NOT ACHIEVED'}")
    logger.info(f"💾 Models saved in 'models/' directory")
    logger.info(f"📈 Training results saved in 'training_results.json'")
    logger.info(f"{'='*60}")

    return final_accuracy


if __name__ == "__main__":
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    try:
        accuracy = main()

        if accuracy >= 0.99:
            logger.info("🎉 SUCCESS: 99% accuracy target achieved!")
            sys.exit(0)
        else:
            logger.info(
                f"⚠️  WARNING: Target accuracy not reached. Achieved: {accuracy:.4f}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
