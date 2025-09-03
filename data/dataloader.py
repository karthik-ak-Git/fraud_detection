"""
Fraud Detection Data Loader

This module handles loading and preprocessing of fraud detection transaction data.
The dataset consists of daily transaction files in pickle format.
"""

import os
import pickle
import pandas as pd
import numpy as np
import warnings
from typing import Tuple, List, Optional
import glob
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class FraudDataLoader:
    """Data loader for fraud detection dataset"""

    def __init__(self, dataset_path: str = "dataset"):
        self.dataset_path = dataset_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.target_column = None

    def load_pickle_with_compatibility(self, file_path: str) -> pd.DataFrame:
        """
        Load pickle file with compatibility fixes for older pandas versions
        """
        try:
            # Try normal loading first
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            return data
        except (ModuleNotFoundError, KeyError) as e:
            logger.warning(f"Compatibility issue with {file_path}: {e}")
            try:
                # Try with compatibility fixes
                import sys

                # Create compatibility mappings
                compatibility_mappings = {
                    'pandas.core.indexes.numeric': 'pandas.core.indexes.api',
                    'pandas.indexes.numeric': 'pandas.core.indexes.api',
                    'pandas.core.indexes.range': 'pandas.core.indexes.range',
                }

                # Apply mappings
                for old_module, new_module in compatibility_mappings.items():
                    try:
                        if new_module in sys.modules:
                            sys.modules[old_module] = sys.modules[new_module]
                    except:
                        pass

                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                return data
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                # If all else fails, return None and we'll skip this file
                return None

    def load_sample_data(self, num_files: int = 5) -> pd.DataFrame:
        """
        Load a sample of the dataset to understand its structure
        """
        pickle_files = glob.glob(os.path.join(self.dataset_path, "*.pkl"))
        pickle_files = sorted(pickle_files)[:num_files]

        data_frames = []

        for file_path in pickle_files:
            logger.info(f"Loading {file_path}")
            df = self.load_pickle_with_compatibility(file_path)
            if df is not None:
                # Add date column from filename
                date_str = os.path.basename(file_path).replace('.pkl', '')
                df['date'] = pd.to_datetime(date_str)
                data_frames.append(df)
            else:
                logger.warning(f"Skipping {file_path} due to loading error")

        if data_frames:
            combined_df = pd.concat(data_frames, ignore_index=True)
            logger.info(
                f"Loaded {len(combined_df)} records from {len(data_frames)} files")
            return combined_df
        else:
            logger.error("No data could be loaded")
            return None

    def load_all_data(self) -> pd.DataFrame:
        """
        Load all available data files
        """
        pickle_files = glob.glob(os.path.join(self.dataset_path, "*.pkl"))
        pickle_files = sorted(pickle_files)

        data_frames = []

        for file_path in pickle_files:
            logger.info(f"Loading {file_path}")
            df = self.load_pickle_with_compatibility(file_path)
            if df is not None:
                # Add date column from filename
                date_str = os.path.basename(file_path).replace('.pkl', '')
                df['date'] = pd.to_datetime(date_str)
                data_frames.append(df)
            else:
                logger.warning(f"Skipping {file_path} due to loading error")

        if data_frames:
            combined_df = pd.concat(data_frames, ignore_index=True)
            logger.info(
                f"Loaded {len(combined_df)} records from {len(data_frames)} files")
            return combined_df
        else:
            logger.error("No data could be loaded")
            return None

    def create_synthetic_data(self, num_samples: int = 10000) -> pd.DataFrame:
        """
        Create synthetic fraud detection data if original data cannot be loaded
        """
        logger.info(
            f"Creating synthetic fraud detection data with {num_samples} samples")

        np.random.seed(42)

        # Generate synthetic transaction features
        data = {
            'amount': np.random.lognormal(3, 1, num_samples),
            'oldbalanceOrg': np.random.exponential(1000, num_samples),
            'newbalanceOrig': np.random.exponential(1000, num_samples),
            'oldbalanceDest': np.random.exponential(1000, num_samples),
            'newbalanceDest': np.random.exponential(1000, num_samples),
            'hour': np.random.randint(0, 24, num_samples),
            'day_of_week': np.random.randint(0, 7, num_samples),
            'transaction_count_1h': np.random.poisson(3, num_samples),
            'transaction_count_24h': np.random.poisson(20, num_samples),
            'type_CASH_IN': np.random.binomial(1, 0.1, num_samples),
            'type_CASH_OUT': np.random.binomial(1, 0.2, num_samples),
            'type_DEBIT': np.random.binomial(1, 0.3, num_samples),
            'type_PAYMENT': np.random.binomial(1, 0.3, num_samples),
            'type_TRANSFER': np.random.binomial(1, 0.1, num_samples),
        }

        df = pd.DataFrame(data)

        # Create fraud labels based on some rules to make it realistic
        fraud_conditions = (
            (df['amount'] > 200000) |  # Large amounts
            # Balance inconsistency
            (df['oldbalanceOrg'] - df['newbalanceOrig'] != df['amount']) |
            # Late night large transactions
            ((df['hour'] < 6) & (df['amount'] > 50000)) |
            (df['transaction_count_1h'] > 10)  # High frequency
        )

        # Add some randomness to fraud labels
        fraud_prob = np.where(fraud_conditions, 0.3, 0.01)
        df['isFraud'] = np.random.binomial(1, fraud_prob)

        # Ensure we have some fraud cases (about 1-2%)
        if df['isFraud'].sum() / len(df) < 0.005:
            fraud_indices = np.random.choice(
                df.index, size=int(0.01 * len(df)), replace=False)
            df.loc[fraud_indices, 'isFraud'] = 1

        logger.info(
            f"Created synthetic data: {len(df)} samples, {df['isFraud'].sum()} fraud cases ({df['isFraud'].mean()*100:.2f}%)")

        return df

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the data for neural network training
        """
        # Identify target column
        if 'isFraud' in df.columns:
            self.target_column = 'isFraud'
        elif 'Class' in df.columns:
            self.target_column = 'Class'
        else:
            # Look for binary column that could be fraud indicator
            binary_cols = [col for col in df.columns if df[col].nunique() == 2]
            if binary_cols:
                self.target_column = binary_cols[0]
                logger.info(f"Using {self.target_column} as target column")
            else:
                raise ValueError(
                    "No suitable target column found. Please ensure your data has a fraud indicator column.")

        # Separate features and target
        y = df[self.target_column].values

        # Select feature columns (exclude target and non-numeric columns)
        numeric_columns = df.select_dtypes(
            include=[np.number]).columns.tolist()
        if self.target_column in numeric_columns:
            numeric_columns.remove(self.target_column)

        # Remove date column if present
        if 'date' in numeric_columns:
            numeric_columns.remove('date')

        self.feature_columns = numeric_columns
        X = df[self.feature_columns].values

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        logger.info(
            f"Preprocessed data: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        logger.info(f"Class distribution: {np.bincount(y)}")

        return X_scaled, y

    def get_data_splits(self, df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1) -> Tuple:
        """
        Split data into train, validation, and test sets
        """
        X, y = self.preprocess_data(df)

        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Second split: train and val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )

        logger.info(
            f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        return X_train, X_val, X_test, y_train, y_val, y_test


class FraudDataset(Dataset):
    """PyTorch Dataset for fraud detection data"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test,
                        batch_size: int = 256) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training, validation, and testing
    """
    train_dataset = FraudDataset(X_train, y_train)
    val_dataset = FraudDataset(X_val, y_val)
    test_dataset = FraudDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the data loader
    loader = FraudDataLoader("dataset")

    # Try to load sample data first
    print("Attempting to load sample data...")
    try:
        df = loader.load_sample_data(num_files=3)
    except Exception as e:
        print(f"Error loading original data: {e}")
        df = None

    if df is None:
        print("Cannot load original data, creating synthetic data...")
        df = loader.create_synthetic_data(num_samples=10000)

    if df is not None:
        print(f"Data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Data types:\n{df.dtypes}")
        print(f"First few rows:\n{df.head()}")

        # Test preprocessing
        try:
            X_train, X_val, X_test, y_train, y_val, y_test = loader.get_data_splits(
                df)
            print(f"✓ Data preprocessing successful!")
            print(f"Feature columns: {loader.feature_columns}")
            print(f"Target column: {loader.target_column}")
        except Exception as e:
            print(f"✗ Data preprocessing failed: {e}")
    else:
        print("✗ Failed to load any data")
