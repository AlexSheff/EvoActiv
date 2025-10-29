"""
Dataset loader module for EvoActiv.
Provides utilities for loading and preprocessing datasets.
"""

import torch
import torchvision
import numpy as np
from typing import Tuple, Dict, Any, Optional
from torch.utils.data import DataLoader, Dataset, random_split


def load_mnist(batch_size: int = 64, 
               train_ratio: float = 0.8,
               data_dir: str = "../data") -> Tuple[DataLoader, DataLoader]:
    """
    Load the MNIST dataset and create train/test dataloaders.
    
    Args:
        batch_size: Batch size for dataloaders
        train_ratio: Ratio of data to use for training
        data_dir: Directory to store the dataset
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Define transforms
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load the dataset
    full_dataset = torchvision.datasets.MNIST(
        root=data_dir, 
        train=True,
        download=True,
        transform=transform
    )
    
    # Split into train and validation sets
    train_size = int(len(full_dataset) * train_ratio)
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def load_custom_dataset(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """
    Load a custom dataset based on configuration.
    
    Args:
        config: Configuration dictionary with dataset parameters
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    dataset_type = config.get("dataset_type", "csv")
    
    if dataset_type == "csv":
        return _load_csv_dataset(config)
    elif dataset_type == "numpy":
        return _load_numpy_dataset(config)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def _load_csv_dataset(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Load dataset from CSV file."""
    import pandas as pd
    
    file_path = config.get("file_path")
    target_column = config.get("target_column")
    feature_columns = config.get("feature_columns")
    batch_size = config.get("batch_size", 32)
    train_ratio = config.get("train_ratio", 0.8)
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Extract features and target
    X = df[feature_columns].values
    y = df[target_column].values
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    
    # Split into train and test
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def _load_numpy_dataset(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Load dataset from NumPy arrays."""
    X_path = config.get("X_path")
    y_path = config.get("y_path")
    batch_size = config.get("batch_size", 32)
    train_ratio = config.get("train_ratio", 0.8)
    
    # Load data
    X = np.load(X_path)
    y = np.load(y_path)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    
    # Split into train and test
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def _load_json_dataset(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    import pandas as pd
    file_path = config.get("file_path") or config.get("custom", {}).get("file_path")
    target_column = config.get("target_column") or config.get("custom", {}).get("target_column")
    feature_columns = config.get("feature_columns") or config.get("custom", {}).get("feature_columns")
    batch_size = config.get("batch_size", 32)
    train_ratio = config.get("train_ratio", 0.8)
    df = pd.read_json(file_path)
    X = df[feature_columns].values
    y = df[target_column].values
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y) if np.issubdtype(y.dtype, np.integer) else torch.FloatTensor(y)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def load_dataset_from_config(dataset_cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    dtype = dataset_cfg.get('type', 'mnist')
    if dtype == 'mnist':
        return load_mnist(dataset_cfg.get('batch_size', 64), dataset_cfg.get('train_ratio', 0.8), dataset_cfg.get('data_dir', '../data'))
    elif dtype in ('custom_csv', 'custom_json', 'custom_numpy', 'csv', 'json', 'numpy', 'auto', 'torchvision'):
        return load_custom_dataset(dataset_cfg)
    else:
        raise ValueError(f"Unknown dataset type: {dtype}")