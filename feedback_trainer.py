"""
Feedback-based Model Trainer

This script handles retraining the fraud detection model using user feedback data.
"""

from main import FraudTrainer
from models.fraud_models import get_model
from data.dataloader import FraudDataLoader, FraudDataset, create_data_loaders
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeedbackTrainer:
    """Trainer for incorporating user feedback into model improvement"""

    def __init__(self, model_path: str, feedback_file: str = "outputs/feedback_log.jsonl"):
        self.model_path = model_path
        self.feedback_file = feedback_file
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.model_info = None
        self.scaler = None

    def load_existing_model(self):
        """Load the existing trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Load model checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)

        self.model_info = {
            'model_type': checkpoint['model_type'],
            'model_params': checkpoint['model_params'],
            'input_size': checkpoint['input_size'],
            'feature_columns': checkpoint['feature_columns']
        }

        # Recreate model
        self.model = get_model(
            self.model_info['model_type'],
            self.model_info['input_size'],
            **self.model_info['model_params']
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)

        logger.info(f"Loaded existing model: {self.model_info['model_type']}")

    def load_feedback_data(self):
        """Load and process feedback data"""
        if not os.path.exists(self.feedback_file):
            logger.warning(f"No feedback file found: {self.feedback_file}")
            return None, None

        feedback_records = []

        # Read JSONL file
        with open(self.feedback_file, 'r') as f:
            for line in f:
                if line.strip():
                    feedback_records.append(json.loads(line))

        if not feedback_records:
            logger.warning("No feedback records found")
            return None, None

        logger.info(f"Loaded {len(feedback_records)} feedback records")

        # Process feedback into training data
        X_feedback = []
        y_feedback = []

        for record in feedback_records:
            try:
                # Extract transaction features
                transaction_data = record['transaction_data']
                feature_vector = []

                for feature in self.model_info['feature_columns']:
                    feature_vector.append(transaction_data.get(feature, 0.0))

                X_feedback.append(feature_vector)

                # Convert actual class to label
                actual_class = record['actual_class']
                label = 1 if actual_class == 'fraud' else 0
                y_feedback.append(label)

            except Exception as e:
                logger.warning(f"Skipping invalid feedback record: {e}")
                continue

        if not X_feedback:
            logger.warning("No valid feedback data found")
            return None, None

        X_feedback = np.array(X_feedback)
        y_feedback = np.array(y_feedback)

        logger.info(f"Processed feedback data: {X_feedback.shape[0]} samples")
        logger.info(f"Feedback class distribution: {np.bincount(y_feedback)}")

        return X_feedback, y_feedback

    def prepare_combined_data(self, feedback_ratio: float = 0.3):
        """Combine original training data with feedback data"""
        # Load feedback data
        X_feedback, y_feedback = self.load_feedback_data()

        if X_feedback is None:
            logger.error("No feedback data available for retraining")
            return None

        # Generate synthetic data for base training
        loader = FraudDataLoader()
        df = loader.create_synthetic_data(num_samples=10000)
        X_synthetic, y_synthetic = loader.preprocess_data(df)

        # Store scaler for later use
        self.scaler = loader.scaler

        # Scale feedback data using the same scaler
        X_feedback_scaled = self.scaler.transform(X_feedback)

        # Determine how much synthetic data to use
        feedback_size = len(X_feedback)
        synthetic_size = int(
            feedback_size * (1 - feedback_ratio) / feedback_ratio)
        synthetic_size = min(synthetic_size, len(X_synthetic))

        # Combine datasets
        X_combined = np.vstack(
            [X_synthetic[:synthetic_size], X_feedback_scaled])
        y_combined = np.hstack([y_synthetic[:synthetic_size], y_feedback])

        logger.info(f"Combined dataset: {len(X_combined)} samples")
        logger.info(f"  - Synthetic: {synthetic_size} samples")
        logger.info(f"  - Feedback: {feedback_size} samples")
        logger.info(f"Combined class distribution: {np.bincount(y_combined)}")

        return X_combined, y_combined

    def fine_tune_model(self, X_data, y_data, epochs: int = 20, learning_rate: float = 0.0001):
        """Fine-tune the model with feedback data"""
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
        )

        # Create data loaders
        train_loader, val_loader, _ = create_data_loaders(
            X_train, X_val, X_val, y_train, y_val, y_val, batch_size=64
        )

        # Setup optimizer with lower learning rate for fine-tuning
        optimizer = optim.Adam(self.model.parameters(),
                               lr=learning_rate, weight_decay=1e-5)

        # Use weighted loss for imbalanced data
        class_counts = np.bincount(y_train)
        class_weights = len(y_train) / (len(class_counts) * class_counts)
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        logger.info(f"Starting fine-tuning for {epochs} epochs...")

        best_val_accuracy = 0
        train_history = {'loss': [], 'accuracy': [],
                         'val_loss': [], 'val_accuracy': []}

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

            train_loss = total_loss / len(train_loader)
            train_accuracy = 100. * correct / total

            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    val_loss += criterion(output, target).item()

                    pred = output.argmax(dim=1, keepdim=True)
                    val_correct += pred.eq(target.view_as(pred)).sum().item()
                    val_total += target.size(0)

            val_loss /= len(val_loader)
            val_accuracy = 100. * val_correct / val_total

            # Update history
            train_history['loss'].append(train_loss)
            train_history['accuracy'].append(train_accuracy)
            train_history['val_loss'].append(val_loss)
            train_history['val_accuracy'].append(val_accuracy)

            logger.info(f"Epoch {epoch+1}/{epochs}:")
            logger.info(
                f"  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
            logger.info(
                f"  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.save_improved_model(val_accuracy, epoch)

        logger.info(
            f"Fine-tuning completed. Best validation accuracy: {best_val_accuracy:.2f}%")
        return train_history

    def save_improved_model(self, val_accuracy: float, epoch: int):
        """Save the improved model"""
        # Create new model path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_model_path = f"outputs/improved_model_feedback_{timestamp}.pth"

        # Save model checkpoint
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epoch': epoch,
            'val_accuracy': val_accuracy,
            'model_type': self.model_info['model_type'],
            'model_params': self.model_info['model_params'],
            'input_size': self.model_info['input_size'],
            'feature_columns': self.model_info['feature_columns'],
            'improvement_type': 'feedback_training',
            'timestamp': timestamp
        }, new_model_path)

        logger.info(f"Improved model saved: {new_model_path}")

        # Update the main model file if improvement is significant
        original_accuracy = 45.62  # From our training results
        if val_accuracy > original_accuracy + 2:  # At least 2% improvement
            best_model_path = "outputs/best_model_nn.pth"
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'epoch': epoch,
                'val_accuracy': val_accuracy,
                'model_type': self.model_info['model_type'],
                'model_params': self.model_info['model_params'],
                'input_size': self.model_info['input_size'],
                'feature_columns': self.model_info['feature_columns'],
                'improvement_type': 'feedback_training',
                'timestamp': timestamp
            }, best_model_path)

            logger.info(
                f"Main model updated with {val_accuracy:.2f}% accuracy (improvement of {val_accuracy - original_accuracy:.2f}%)")

    def evaluate_improvement(self):
        """Evaluate the model improvement using feedback data"""
        X_feedback, y_feedback = self.load_feedback_data()

        if X_feedback is None:
            logger.warning("No feedback data for evaluation")
            return

        # Scale feedback data
        X_feedback_scaled = self.scaler.transform(X_feedback)

        # Make predictions
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_feedback_scaled).to(self.device)
            outputs = self.model(X_tensor)
            predictions = outputs.argmax(dim=1).cpu().numpy()

        # Calculate accuracy on feedback data
        accuracy = accuracy_score(y_feedback, predictions)

        logger.info(f"Model accuracy on feedback data: {accuracy:.4f}")
        logger.info("Detailed classification report on feedback data:")
        logger.info(classification_report(y_feedback, predictions,
                                          target_names=['Normal', 'Fraud']))

        return accuracy

    def retrain_with_feedback(self, epochs: int = 20, feedback_ratio: float = 0.3):
        """Complete retraining process with feedback data"""
        logger.info("Starting feedback-based model improvement...")

        # Load existing model
        self.load_existing_model()

        # Prepare combined data
        X_combined, y_combined = self.prepare_combined_data(feedback_ratio)

        if X_combined is None:
            logger.error("Cannot proceed without feedback data")
            return

        # Fine-tune model
        history = self.fine_tune_model(X_combined, y_combined, epochs)

        # Evaluate improvement
        self.evaluate_improvement()

        logger.info("Feedback-based improvement completed!")
        return history


def main():
    """Main function for feedback training"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Retrain fraud detection model with feedback')
    parser.add_argument('--model_path', type=str, default='outputs/best_model_nn.pth',
                        help='Path to existing model')
    parser.add_argument('--feedback_file', type=str, default='outputs/feedback_log.jsonl',
                        help='Path to feedback log file')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--feedback_ratio', type=float, default=0.3,
                        help='Ratio of feedback data in training set')

    args = parser.parse_args()

    # Check if model and feedback files exist
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        return

    if not os.path.exists(args.feedback_file):
        logger.error(f"Feedback file not found: {args.feedback_file}")
        logger.info("Collect some feedback through the web interface first")
        return

    # Initialize feedback trainer
    trainer = FeedbackTrainer(args.model_path, args.feedback_file)

    # Retrain with feedback
    trainer.retrain_with_feedback(args.epochs, args.feedback_ratio)


if __name__ == "__main__":
    main()
