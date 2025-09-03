"""
Utility script for common fraud detection system tasks
"""

import os
import sys
import argparse
import subprocess
import json
import pandas as pd
from datetime import datetime


def start_api():
    """Start the fraud detection API server"""
    print("🚀 Starting Fraud Detection API...")
    script_path = os.path.join(os.path.dirname(__file__), "main_api.py")
    python_exe = os.path.join(os.path.dirname(
        __file__), ".venv", "Scripts", "python.exe")

    if os.path.exists(python_exe):
        subprocess.run([python_exe, script_path])
    else:
        subprocess.run(["python", script_path])


def train_model():
    """Train a new fraud detection model"""
    print("🧠 Training fraud detection model...")
    script_path = os.path.join(os.path.dirname(__file__), "main.py")
    python_exe = os.path.join(os.path.dirname(
        __file__), ".venv", "Scripts", "python.exe")

    if os.path.exists(python_exe):
        subprocess.run([python_exe, script_path])
    else:
        subprocess.run(["python", script_path])


def evaluate_model():
    """Evaluate the trained model"""
    print("📊 Evaluating fraud detection model...")
    script_path = os.path.join(os.path.dirname(__file__), "evaluate.py")
    python_exe = os.path.join(os.path.dirname(
        __file__), ".venv", "Scripts", "python.exe")

    if os.path.exists(python_exe):
        subprocess.run([python_exe, script_path])
    else:
        subprocess.run(["python", script_path])


def retrain_with_feedback():
    """Retrain model with user feedback"""
    print("🔄 Retraining model with feedback...")
    script_path = os.path.join(
        os.path.dirname(__file__), "feedback_trainer.py")
    python_exe = os.path.join(os.path.dirname(
        __file__), ".venv", "Scripts", "python.exe")

    if os.path.exists(python_exe):
        subprocess.run([python_exe, script_path])
    else:
        subprocess.run(["python", script_path])


def show_stats():
    """Show system statistics"""
    print("📈 Fraud Detection System Statistics")
    print("=" * 50)

    # Check if outputs directory exists
    outputs_dir = "outputs"
    if not os.path.exists(outputs_dir):
        print("❌ No outputs directory found. Train a model first.")
        return

    # Model information
    model_path = os.path.join(outputs_dir, "best_model_nn.pth")
    if os.path.exists(model_path):
        print("✅ Model Status: Trained")
        stat = os.stat(model_path)
        print(f"📅 Last Training: {datetime.fromtimestamp(stat.st_mtime)}")
    else:
        print("❌ Model Status: Not trained")
        return

    # Evaluation metrics
    metrics_path = os.path.join(outputs_dir, "evaluation_metrics_nn.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        print("\\n🎯 Model Performance:")
        print(f"   Accuracy: {metrics.get('accuracy', 0):.1%}")
        print(f"   ROC AUC: {metrics.get('roc_auc', 0):.1%}")
        print(f"   Fraud Precision: {metrics.get('precision_fraud', 0):.1%}")
        print(f"   Fraud Recall: {metrics.get('recall_fraud', 0):.1%}")
        print(f"   Fraud F1-Score: {metrics.get('f1_fraud', 0):.1%}")

    # Feedback information
    feedback_path = os.path.join(outputs_dir, "feedback_log.jsonl")
    if os.path.exists(feedback_path):
        feedback_count = 0
        correct_predictions = 0

        with open(feedback_path, 'r') as f:
            for line in f:
                if line.strip():
                    feedback = json.loads(line)
                    feedback_count += 1
                    if feedback.get('predicted_class') == feedback.get('actual_class'):
                        correct_predictions += 1

        print(f"\\n💬 User Feedback:")
        print(f"   Total Feedback: {feedback_count}")
        print(f"   Correct Predictions: {correct_predictions}")
        if feedback_count > 0:
            accuracy = correct_predictions / feedback_count * 100
            print(f"   Feedback Accuracy: {accuracy:.1f}%")
    else:
        print("\\n💬 User Feedback: No feedback collected yet")

    print("\\n📁 Output Files:")
    for file in os.listdir(outputs_dir):
        file_path = os.path.join(outputs_dir, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            if size > 1024 * 1024:
                size_str = f"{size / (1024 * 1024):.1f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"   📄 {file} ({size_str})")


def setup_environment():
    """Setup the development environment"""
    print("🔧 Setting up fraud detection environment...")

    # Check Python version
    import sys
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return

    print(f"✅ Python {sys.version.split()[0]} detected")

    # Check virtual environment
    venv_path = ".venv"
    if not os.path.exists(venv_path):
        print("📦 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", venv_path])
    else:
        print("✅ Virtual environment exists")

    # Install requirements
    requirements_path = "requirements.txt"
    if os.path.exists(requirements_path):
        print("📦 Installing requirements...")
        python_exe = os.path.join(venv_path, "Scripts", "python.exe")
        if os.path.exists(python_exe):
            subprocess.run(
                [python_exe, "-m", "pip", "install", "-r", requirements_path])
        else:
            print("❌ Virtual environment Python not found")
    else:
        print("❌ requirements.txt not found")

    print("✅ Environment setup complete!")


def open_frontend():
    """Open the frontend interface"""
    frontend_path = os.path.join("frontend", "index.html")
    if os.path.exists(frontend_path):
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(frontend_path)}")
        print("🌐 Frontend opened in browser")
    else:
        print("❌ Frontend files not found")


def main():
    parser = argparse.ArgumentParser(
        description="Fraud Detection System Utility")
    subparsers = parser.add_subparsers(
        dest='command', help='Available commands')

    # Add subcommands
    subparsers.add_parser('setup', help='Setup development environment')
    subparsers.add_parser('train', help='Train a new model')
    subparsers.add_parser('evaluate', help='Evaluate the trained model')
    subparsers.add_parser('api', help='Start the API server')
    subparsers.add_parser('retrain', help='Retrain model with feedback')
    subparsers.add_parser('stats', help='Show system statistics')
    subparsers.add_parser('frontend', help='Open frontend interface')

    args = parser.parse_args()

    if args.command == 'setup':
        setup_environment()
    elif args.command == 'train':
        train_model()
    elif args.command == 'evaluate':
        evaluate_model()
    elif args.command == 'api':
        start_api()
    elif args.command == 'retrain':
        retrain_with_feedback()
    elif args.command == 'stats':
        show_stats()
    elif args.command == 'frontend':
        open_frontend()
    else:
        # Show help menu
        print("🛡️ Fraud Detection System Utility")
        print("=" * 40)
        print("Available commands:")
        print("  python utils.py setup     - Setup environment")
        print("  python utils.py train     - Train new model")
        print("  python utils.py evaluate  - Evaluate model")
        print("  python utils.py api       - Start API server")
        print("  python utils.py retrain   - Retrain with feedback")
        print("  python utils.py stats     - Show statistics")
        print("  python utils.py frontend  - Open web interface")
        print("\\nExample usage:")
        print("  python utils.py setup")
        print("  python utils.py train")
        print("  python utils.py api")


if __name__ == "__main__":
    main()
