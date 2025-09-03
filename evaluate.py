"""
Fraud Detection Model Evaluation Script

This script evaluates trained fraud detection models and provides detailed analysis.
"""

from models.fraud_models import get_model
from data.dataloader import FraudDataLoader, create_data_loaders
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve, precision_recall_curve
)
import json
import argparse
from datetime import datetime
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudEvaluator:
    """Comprehensive evaluator for fraud detection models"""

    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the evaluator

        Args:
            model_path: Path to the saved model
            device: Device to run evaluation on
        """
        self.model_path = model_path

        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = None
        self.model_info = None
        self.test_loader = None

    def load_model(self):
        """Load the trained model"""
        checkpoint = torch.load(self.model_path, map_location=self.device)

        self.model_info = {
            'model_type': checkpoint['model_type'],
            'model_params': checkpoint['model_params'],
            'input_size': checkpoint['input_size'],
            'feature_columns': checkpoint['feature_columns'],
            'epoch': checkpoint['epoch'],
            'val_loss': checkpoint['val_loss'],
            'val_accuracy': checkpoint['val_accuracy']
        }

        # Recreate model
        self.model = get_model(
            self.model_info['model_type'],
            self.model_info['input_size'],
            **self.model_info['model_params']
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info(f"Loaded model: {self.model_info['model_type']}")
        logger.info(f"Model trained for {self.model_info['epoch']} epochs")
        logger.info(
            f"Validation accuracy: {self.model_info['val_accuracy']:.2f}%")

    def prepare_test_data(self, data_path: str = "dataset", use_synthetic: bool = True):
        """Prepare test data for evaluation"""
        loader = FraudDataLoader(data_path)

        if use_synthetic:
            df = loader.create_synthetic_data(num_samples=20000)
        else:
            df = loader.load_sample_data(num_files=5)
            if df is None:
                df = loader.create_synthetic_data(num_samples=20000)

        # Get data splits
        X_train, X_val, X_test, y_train, y_val, y_test = loader.get_data_splits(
            df)

        # Create test data loader
        _, _, test_loader = create_data_loaders(
            X_train, X_val, X_test, y_train, y_val, y_test, batch_size=256
        )

        self.test_loader = test_loader
        logger.info(f"Test data prepared: {len(test_loader.dataset)} samples")

    def predict(self, return_probabilities: bool = True):
        """Make predictions on test data"""
        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                probabilities = F.softmax(output, dim=1)
                predictions = output.argmax(dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        results = {
            'predictions': np.array(all_predictions),
            'targets': np.array(all_targets),
            'probabilities': np.array(all_probabilities)
        }

        return results

    def calculate_metrics(self, results: dict):
        """Calculate comprehensive evaluation metrics"""
        y_true = results['targets']
        y_pred = results['predictions']
        y_prob = results['probabilities'][:, 1]  # Probability of fraud class

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_true, y_prob),

            # Per-class metrics
            'precision_normal': precision_score(y_true, y_pred, pos_label=0),
            'recall_normal': recall_score(y_true, y_pred, pos_label=0),
            'f1_normal': f1_score(y_true, y_pred, pos_label=0),

            'precision_fraud': precision_score(y_true, y_pred, pos_label=1),
            'recall_fraud': recall_score(y_true, y_pred, pos_label=1),
            'f1_fraud': f1_score(y_true, y_pred, pos_label=1),
        }

        return metrics

    def generate_detailed_report(self, results: dict, save_path: str = "outputs"):
        """Generate detailed evaluation report with visualizations"""
        os.makedirs(save_path, exist_ok=True)

        y_true = results['targets']
        y_pred = results['predictions']
        y_prob = results['probabilities']

        # Calculate metrics
        metrics = self.calculate_metrics(results)

        # Generate report
        report_path = os.path.join(
            save_path, f"detailed_evaluation_{self.model_info['model_type']}.html")

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fraud Detection Model Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
                .section {{ margin: 20px 0; }}
                .metric {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #007bff; }}
                .metric-value {{ font-size: 1.5em; font-weight: bold; color: #007bff; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                th {{ background-color: #f2f2f2; }}
                .fraud {{ background-color: #ffebee; }}
                .normal {{ background-color: #e8f5e8; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Fraud Detection Model Evaluation Report</h1>
                <p>Model: {self.model_info['model_type'].upper()} | Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Model Information</h2>
                <p><strong>Model Type:</strong> {self.model_info['model_type']}</p>
                <p><strong>Input Features:</strong> {self.model_info['input_size']}</p>
                <p><strong>Training Epochs:</strong> {self.model_info['epoch']}</p>
                <p><strong>Validation Accuracy:</strong> {self.model_info['val_accuracy']:.2f}%</p>
            </div>
            
            <div class="section">
                <h2>Overall Performance Metrics</h2>
                <div class="metric">
                    <strong>Accuracy:</strong> <span class="metric-value">{metrics['accuracy']:.4f}</span>
                </div>
                <div class="metric">
                    <strong>ROC AUC:</strong> <span class="metric-value">{metrics['roc_auc']:.4f}</span>
                </div>
                <div class="metric">
                    <strong>F1 Score (Weighted):</strong> <span class="metric-value">{metrics['f1_score']:.4f}</span>
                </div>
                <div class="metric">
                    <strong>Precision (Weighted):</strong> <span class="metric-value">{metrics['precision']:.4f}</span>
                </div>
                <div class="metric">
                    <strong>Recall (Weighted):</strong> <span class="metric-value">{metrics['recall']:.4f}</span>
                </div>
            </div>
            
            <div class="section">
                <h2>Per-Class Performance</h2>
                <table>
                    <tr>
                        <th>Class</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                    </tr>
                    <tr class="normal">
                        <td><strong>Normal Transactions</strong></td>
                        <td>{metrics['precision_normal']:.4f}</td>
                        <td>{metrics['recall_normal']:.4f}</td>
                        <td>{metrics['f1_normal']:.4f}</td>
                    </tr>
                    <tr class="fraud">
                        <td><strong>Fraud Transactions</strong></td>
                        <td>{metrics['precision_fraud']:.4f}</td>
                        <td>{metrics['recall_fraud']:.4f}</td>
                        <td>{metrics['f1_fraud']:.4f}</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Data Distribution</h2>
                <p><strong>Total Test Samples:</strong> {len(y_true)}</p>
                <p><strong>Normal Transactions:</strong> {np.sum(y_true == 0)} ({100 * np.sum(y_true == 0) / len(y_true):.1f}%)</p>
                <p><strong>Fraud Transactions:</strong> {np.sum(y_true == 1)} ({100 * np.sum(y_true == 1) / len(y_true):.1f}%)</p>
            </div>
            
            <div class="section">
                <h2>Feature Importance</h2>
                <p>Model features (in order of input):</p>
                <ol>
        """

        for feature in self.model_info['feature_columns']:
            html_content += f"<li>{feature}</li>"

        html_content += """
                </ol>
            </div>
        </body>
        </html>
        """

        with open(report_path, 'w') as f:
            f.write(html_content)

        logger.info(f"Detailed report saved to {report_path}")

        return metrics

    def plot_advanced_metrics(self, results: dict, save_path: str = "outputs"):
        """Create advanced visualization plots"""
        y_true = results['targets']
        y_pred = results['predictions']
        y_prob = results['probabilities']

        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Advanced Fraud Detection Analysis - {self.model_info["model_type"].upper()}',
                     fontsize=16, fontweight='bold')

        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                    xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = roc_auc_score(y_true, y_prob[:, 1])
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2,
                        label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend(loc="lower right")
        axes[0, 1].grid(True)

        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
        axes[0, 2].plot(recall, precision, color='blue', lw=2)
        axes[0, 2].set_xlabel('Recall')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].set_title('Precision-Recall Curve')
        axes[0, 2].grid(True)

        # 4. Prediction Probability Distribution
        fraud_probs = y_prob[y_true == 1, 1]
        normal_probs = y_prob[y_true == 0, 1]

        axes[1, 0].hist(normal_probs, bins=50, alpha=0.7,
                        label='Normal', color='green', density=True)
        axes[1, 0].hist(fraud_probs, bins=50, alpha=0.7,
                        label='Fraud', color='red', density=True)
        axes[1, 0].set_xlabel('Fraud Probability')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Prediction Probability Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # 5. Threshold Analysis
        thresholds = np.arange(0.1, 1.0, 0.05)
        f1_scores = []
        precisions = []
        recalls = []

        for threshold in thresholds:
            pred_thresh = (y_prob[:, 1] >= threshold).astype(int)
            f1_scores.append(f1_score(y_true, pred_thresh))
            precisions.append(precision_score(
                y_true, pred_thresh, zero_division=0))
            recalls.append(recall_score(y_true, pred_thresh, zero_division=0))

        axes[1, 1].plot(thresholds, f1_scores, label='F1 Score', marker='o')
        axes[1, 1].plot(thresholds, precisions, label='Precision', marker='s')
        axes[1, 1].plot(thresholds, recalls, label='Recall', marker='^')
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Threshold Analysis')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        # 6. Class Distribution
        class_counts = pd.Series(y_true).value_counts()
        pred_counts = pd.Series(y_pred).value_counts()

        x = ['Normal', 'Fraud']
        width = 0.35
        x_pos = np.arange(len(x))

        axes[1, 2].bar(x_pos - width/2, [class_counts[0], class_counts[1]],
                       width, label='True', alpha=0.8, color='lightblue')
        axes[1, 2].bar(x_pos + width/2, [pred_counts.get(0, 0), pred_counts.get(1, 0)],
                       width, label='Predicted', alpha=0.8, color='lightcoral')

        axes[1, 2].set_xlabel('Class')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title('Class Distribution Comparison')
        axes[1, 2].set_xticks(x_pos)
        axes[1, 2].set_xticklabels(x)
        axes[1, 2].legend()
        axes[1, 2].grid(True, axis='y')

        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(
            save_path, f"advanced_analysis_{self.model_info['model_type']}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Advanced analysis plot saved to {plot_path}")

    def evaluate_comprehensive(self, data_path: str = "dataset",
                               use_synthetic: bool = True,
                               save_path: str = "outputs"):
        """Run comprehensive evaluation"""
        logger.info("Starting comprehensive evaluation...")

        # Load model and prepare data
        self.load_model()
        self.prepare_test_data(data_path, use_synthetic)

        # Make predictions
        results = self.predict()

        # Calculate metrics and generate reports
        metrics = self.generate_detailed_report(results, save_path)
        self.plot_advanced_metrics(results, save_path)

        # Save metrics to JSON
        metrics_path = os.path.join(
            save_path, f"comprehensive_metrics_{self.model_info['model_type']}.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Print summary
        logger.info("Evaluation Summary:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"  Fraud Detection F1: {metrics['f1_fraud']:.4f}")
        logger.info(
            f"  Fraud Detection Precision: {metrics['precision_fraud']:.4f}")
        logger.info(f"  Fraud Detection Recall: {metrics['recall_fraud']:.4f}")

        return metrics


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(
        description='Evaluate Fraud Detection Model')
    parser.add_argument('--model_path', type=str, default='outputs/best_model_nn.pth',
                        help='Path to saved model')
    parser.add_argument('--data_path', type=str, default='dataset',
                        help='Path to data directory')
    parser.add_argument('--save_path', type=str, default='outputs',
                        help='Path to save evaluation results')
    parser.add_argument('--synthetic', action='store_true', default=True,
                        help='Use synthetic data for evaluation')

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        return

    # Initialize evaluator
    evaluator = FraudEvaluator(args.model_path)

    # Run comprehensive evaluation
    evaluator.evaluate_comprehensive(
        data_path=args.data_path,
        use_synthetic=args.synthetic,
        save_path=args.save_path
    )

    logger.info("Comprehensive evaluation completed!")


if __name__ == "__main__":
    main()
