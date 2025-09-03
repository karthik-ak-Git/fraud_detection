"""
Enhanced Data Loader for 99% Accuracy

Advanced data preprocessing and augmentation techniques for superior model performance.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import logging
from typing import Tuple, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedFraudDataLoader:
    """Enhanced data loader with advanced preprocessing for 99% accuracy"""

    def __init__(self, dataset_path: str = "dataset"):
        self.dataset_path = dataset_path
        self.standard_scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.power_transformer = PowerTransformer(method='yeo-johnson')
        self.feature_selector = SelectKBest(f_classif, k='all')
        self.feature_columns = []
        self.target_column = None

    def create_high_quality_synthetic_data(self, num_samples: int = 100000) -> pd.DataFrame:
        """
        Create high-quality synthetic fraud detection data with realistic patterns
        """
        logger.info(
            f"Creating high-quality synthetic data with {num_samples} samples")

        np.random.seed(42)

        # Create more realistic transaction patterns
        data = {}

        # Transaction amounts with realistic distribution
        # Normal transactions: mostly small amounts
        # Fraud transactions: bimodal (very small or very large)
        normal_mask = np.random.random(num_samples) > 0.02  # 2% fraud rate

        amounts = np.zeros(num_samples)
        amounts[normal_mask] = np.random.lognormal(
            mean=3, sigma=1, size=np.sum(normal_mask))
        amounts[~normal_mask] = np.concatenate([
            np.random.lognormal(mean=1, sigma=0.5, size=np.sum(
                ~normal_mask)//2),  # Small fraud
            np.random.lognormal(mean=6, sigma=0.8, size=np.sum(
                ~normal_mask) - np.sum(~normal_mask)//2)  # Large fraud
        ])
        data['amount'] = amounts

        # Account balances with realistic relationships
        # Origin account balances
        data['oldbalanceOrg'] = np.random.exponential(
            scale=2000, size=num_samples)

        # New balance should reflect transaction
        balance_change_noise = np.random.normal(0, 50, num_samples)
        data['newbalanceOrig'] = np.maximum(
            0, data['oldbalanceOrg'] - data['amount'] + balance_change_noise)

        # Destination balances
        data['oldbalanceDest'] = np.random.exponential(
            scale=1500, size=num_samples)
        dest_change_noise = np.random.normal(0, 30, num_samples)
        data['newbalanceDest'] = data['oldbalanceDest'] + \
            data['amount'] + dest_change_noise

        # Time features with fraud patterns
        data['hour'] = np.random.randint(0, 24, num_samples)
        data['day_of_week'] = np.random.randint(0, 7, num_samples)

        # Transaction frequency (higher for fraud)
        data['transaction_count_1h'] = np.random.poisson(
            lam=2, size=num_samples)
        data['transaction_count_24h'] = np.random.poisson(
            lam=15, size=num_samples)

        # Increase frequency for potential fraud cases
        fraud_indices = np.where(~normal_mask)[0]
        data['transaction_count_1h'][fraud_indices] += np.random.poisson(
            lam=5, size=len(fraud_indices))
        data['transaction_count_24h'][fraud_indices] += np.random.poisson(
            lam=20, size=len(fraud_indices))

        # Transaction types with realistic distribution
        transaction_types = ['CASH_IN', 'CASH_OUT',
                             'DEBIT', 'PAYMENT', 'TRANSFER']
        type_probs = [0.05, 0.25, 0.15, 0.45, 0.10]

        chosen_types = np.random.choice(
            transaction_types, size=num_samples, p=type_probs)

        for ttype in transaction_types:
            data[f'type_{ttype}'] = (chosen_types == ttype).astype(int)

        # Additional engineered features for better discrimination
        data['amount_to_balance_ratio'] = data['amount'] / \
            (data['oldbalanceOrg'] + 1)
        data['balance_change_orig'] = data['oldbalanceOrg'] - \
            data['newbalanceOrig']
        data['balance_change_dest'] = data['newbalanceDest'] - \
            data['oldbalanceDest']
        data['balance_inconsistency'] = np.abs(
            data['balance_change_orig'] - data['amount'])

        # Velocity features
        data['transaction_velocity'] = data['transaction_count_1h'] / 1.0
        data['daily_velocity'] = data['transaction_count_24h'] / 24.0

        # Risk indicators
        data['late_night_transaction'] = (
            (data['hour'] >= 23) | (data['hour'] <= 5)).astype(int)
        data['weekend_transaction'] = (data['day_of_week'] >= 5).astype(int)
        data['round_amount'] = (data['amount'] % 100 == 0).astype(int)

        # Merchant/location features (synthetic)
        data['merchant_risk_score'] = np.random.beta(2, 5, num_samples) * 10
        data['location_risk_score'] = np.random.beta(2, 8, num_samples) * 10

        # Device/channel features
        data['mobile_transaction'] = np.random.binomial(1, 0.6, num_samples)
        data['new_device'] = np.random.binomial(1, 0.1, num_samples)

        df = pd.DataFrame(data)

        # Create sophisticated fraud labels
        fraud_score = (
            # Amount-based risk
            (df['amount'] > df['amount'].quantile(0.98)) * 3 +
            (df['amount'] < 1) * 2 +

            # Balance inconsistency
            (df['balance_inconsistency'] > 100) * 4 +

            # Time-based risk
            df['late_night_transaction'] * 2 +
            df['weekend_transaction'] * 1 +

            # Velocity risk
            (df['transaction_count_1h'] > 10) * 3 +
            (df['transaction_count_24h'] > 50) * 2 +

            # Ratio-based risk
            (df['amount_to_balance_ratio'] > 0.9) * 3 +

            # Additional risk factors
            (df['merchant_risk_score'] > 8) * 2 +
            (df['location_risk_score'] > 8) * 2 +
            df['new_device'] * 2 +
            (df['round_amount'] & (df['amount'] > 10000)) * 2 +

            # Transaction type risk
            df['type_CASH_OUT'] * 2 +
            df['type_TRANSFER'] * 1
        )

        # Add some randomness and ensure we have the right fraud rate
        fraud_threshold = np.percentile(fraud_score, 98)  # Top 2% as fraud
        df['isFraud'] = (fraud_score >= fraud_threshold).astype(int)

        # Add some noise to make it more realistic
        noise_fraud = np.random.choice(
            df.index, size=int(0.001 * len(df)), replace=False)
        df.loc[noise_fraud, 'isFraud'] = 1 - df.loc[noise_fraud, 'isFraud']

        logger.info(
            f"Created synthetic data: {len(df)} samples, {df['isFraud'].sum()} fraud cases ({df['isFraud'].mean()*100:.2f}%)")

        return df

    def advanced_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply advanced feature engineering techniques
        """
        logger.info("Applying advanced feature engineering...")

        # Polynomial features for important pairs
        df['amount_squared'] = df['amount'] ** 2
        df['amount_log'] = np.log1p(df['amount'])
        df['balance_product'] = df['oldbalanceOrg'] * df['oldbalanceDest']

        # Interaction features
        df['amount_hour_interaction'] = df['amount'] * df['hour']
        df['velocity_amount_interaction'] = df['transaction_count_1h'] * df['amount']

        # Binned features
        df['amount_bin'] = pd.qcut(
            df['amount'], q=10, labels=False, duplicates='drop')
        df['hour_bin'] = pd.cut(df['hour'], bins=4, labels=False)

        # Statistical features
        amount_mean = df['amount'].mean()
        amount_std = df['amount'].std()
        df['amount_zscore'] = (df['amount'] - amount_mean) / amount_std

        # Trend features
        df['balance_trend'] = (df['newbalanceOrig'] -
                               df['oldbalanceOrg']) / (df['oldbalanceOrg'] + 1)

        return df

    def advanced_preprocessing(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply advanced preprocessing for maximum accuracy
        """
        logger.info("Applying advanced preprocessing...")

        # Feature engineering
        df = self.advanced_feature_engineering(df)

        # Identify target
        if 'isFraud' in df.columns:
            self.target_column = 'isFraud'
        elif 'Class' in df.columns:
            self.target_column = 'Class'
        else:
            raise ValueError("No target column found")

        y = df[self.target_column].values

        # Select numeric features
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
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        # Apply multiple scaling techniques and combine
        X_standard = self.standard_scaler.fit_transform(X)
        X_robust = self.robust_scaler.fit_transform(X)
        X_power = self.power_transformer.fit_transform(X)

        # Combine different scalings (ensemble approach)
        X_combined = np.concatenate([X_standard, X_robust, X_power], axis=1)

        # Feature selection on combined features
        X_selected = self.feature_selector.fit_transform(X_combined, y)

        logger.info(
            f"Preprocessing complete: {X_selected.shape[0]} samples, {X_selected.shape[1]} features")
        logger.info(f"Class distribution: {np.bincount(y)}")

        return X_selected, y

    def create_balanced_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a balanced dataset using advanced sampling techniques
        """
        logger.info("Creating balanced dataset...")

        # Use SMOTETomek for both over and under sampling
        smote_tomek = SMOTETomek(
            smote=SMOTE(sampling_strategy=0.5, random_state=42),
            tomek=TomekLinks(sampling_strategy='majority'),
            random_state=42
        )

        X_balanced, y_balanced = smote_tomek.fit_resample(X, y)

        logger.info(f"Balanced dataset: {X_balanced.shape[0]} samples")
        logger.info(f"Balanced class distribution: {np.bincount(y_balanced)}")

        return X_balanced, y_balanced

    def get_enhanced_data_splits(self, df: pd.DataFrame,
                                 test_size: float = 0.15,
                                 val_size: float = 0.15,
                                 use_balancing: bool = True) -> Tuple:
        """
        Create enhanced data splits with balancing and stratification
        """
        # Apply advanced preprocessing
        X, y = self.advanced_preprocessing(df)

        # Use stratified k-fold for better splits
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Get first split for train/temp
        train_idx, temp_idx = next(skf.split(X, y))
        X_train, X_temp = X[train_idx], X[temp_idx]
        y_train, y_temp = y[train_idx], y[temp_idx]

        # Split temp into val and test
        skf_temp = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        val_idx, test_idx = next(skf_temp.split(X_temp, y_temp))
        X_val, X_test = X_temp[val_idx], X_temp[test_idx]
        y_val, y_test = y_temp[val_idx], y_temp[test_idx]

        # Apply balancing to training data only
        if use_balancing:
            X_train, y_train = self.create_balanced_dataset(X_train, y_train)

        logger.info(
            f"Enhanced data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        return X_train, X_val, X_test, y_train, y_val, y_test


class EnhancedFraudDataset(Dataset):
    """Enhanced PyTorch Dataset with data augmentation"""

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]

        if self.augment and self.training:
            # Add noise augmentation for better generalization
            noise_factor = 0.01
            noise = torch.randn_like(x) * noise_factor
            x = x + noise

        return x, y


def create_enhanced_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test,
                                 batch_size: int = 512, num_workers: int = 0) -> Tuple:
    """
    Create enhanced PyTorch DataLoaders with weighted sampling
    """
    # Create datasets
    train_dataset = EnhancedFraudDataset(X_train, y_train, augment=True)
    val_dataset = EnhancedFraudDataset(X_val, y_val, augment=False)
    test_dataset = EnhancedFraudDataset(X_test, y_test, augment=False)

    # Create weighted sampler for training to handle class imbalance
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights))

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test enhanced data loader
    loader = EnhancedFraudDataLoader("dataset")

    print("Creating high-quality synthetic data...")
    df = loader.create_high_quality_synthetic_data(num_samples=50000)

    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Fraud rate: {df['isFraud'].mean()*100:.2f}%")

    # Test enhanced preprocessing
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = loader.get_enhanced_data_splits(
            df)
        print(f"✅ Enhanced preprocessing successful!")
        print(f"Training features shape: {X_train.shape}")
        print(
            f"Training samples - Normal: {np.sum(y_train == 0)}, Fraud: {np.sum(y_train == 1)}")

        # Test data loaders
        train_loader, val_loader, test_loader = create_enhanced_data_loaders(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        print(f"✅ Enhanced data loaders created successfully!")

    except Exception as e:
        print(f"❌ Enhanced preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
