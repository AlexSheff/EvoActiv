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


def load_torchvision(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Load torchvision dataset (MNIST, FashionMNIST, CIFAR10)."""
    import torchvision.transforms as transforms
    
    name = config.get('name', 'mnist').lower()
    data_dir = config.get('data_dir', './data/torchvision')
    batch_size = config.get('batch_size', 128)
    train_ratio = config.get('train_ratio', 0.8)
    
    # Define transforms based on dataset
    if name in ['mnist', 'fashionmnist']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        raise ValueError(f"Unsupported torchvision dataset: {name}")
    
    # Load dataset
    if name == 'mnist':
        dataset_class = torchvision.datasets.MNIST
    elif name == 'fashionmnist':
        dataset_class = torchvision.datasets.FashionMNIST
    elif name == 'cifar10':
        dataset_class = torchvision.datasets.CIFAR10
    
    full_dataset = dataset_class(root=data_dir, train=True, download=True, transform=transform)
    
    # Split dataset
    train_size = int(len(full_dataset) * train_ratio)
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def load_custom_dataset(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """
    Load a custom dataset based on configuration with auto-detection.
    
    Args:
        config: Configuration dictionary with dataset parameters
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Auto-detect dataset type if not specified
    dataset_type = config.get("dataset_type")
    path = config.get("path")
    
    if not dataset_type and path:
        # Auto-detect based on file extension
        if path.endswith('.csv'):
            dataset_type = "csv"
        elif path.endswith('.json'):
            dataset_type = "json"
        elif path.endswith('.npz'):
            dataset_type = "numpy"
        else:
            dataset_type = "csv"  # default
    elif not dataset_type:
        dataset_type = "csv"  # default
    
    if dataset_type == "csv":
        return _load_csv_dataset(config)
    elif dataset_type == "json":
        return _load_json_dataset(config)
    elif dataset_type == "numpy":
        return _load_numpy_dataset(config)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def _load_csv_dataset(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Load dataset from CSV file."""
    import pandas as pd
    
    file_path = config.get("path") or config.get("file_path")
    target_column = config.get("target_column")
    input_columns = config.get("input_columns") or config.get("feature_columns")
    batch_size = config.get("batch_size", 128)
    train_ratio = config.get("train_ratio", 0.8)
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Extract features and target
    X = df[input_columns].values
    y = df[target_column].values
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y) if np.issubdtype(y.dtype, np.integer) else torch.FloatTensor(y)
    
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
    """Load dataset from NumPy .npz file."""
    file_path = config.get("path") or config.get("file_path")
    batch_size = config.get("batch_size", 128)
    train_ratio = config.get("train_ratio", 0.8)
    
    # Load data from .npz file
    data = np.load(file_path)
    X = data['X']
    y = data['y']
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y) if np.issubdtype(y.dtype, np.integer) else torch.FloatTensor(y)
    
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
    """Load dataset from JSON file."""
    import json
    
    file_path = config.get("path") or config.get("file_path")
    json_input_key = config.get("json_input_key", "inputs")
    json_target_key = config.get("json_target_key", "targets")
    batch_size = config.get("batch_size", 128)
    train_ratio = config.get("train_ratio", 0.8)
    
    # Load JSON data
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract features and targets
    X = np.array(data[json_input_key])
    y = np.array(data[json_target_key])
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y) if np.issubdtype(y.dtype, np.integer) else torch.FloatTensor(y)
    
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


def load_dataset_from_config(dataset_cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    dtype = dataset_cfg.get('type', 'mnist')
    if dtype == 'mnist':
        return load_mnist(dataset_cfg.get('batch_size', 64), dataset_cfg.get('train_ratio', 0.8), dataset_cfg.get('data_dir', '../data'))
    elif dtype == 'torchvision':
        return load_torchvision(dataset_cfg)
    elif dtype in ('custom', 'custom_csv', 'custom_json', 'custom_numpy', 'csv', 'json', 'numpy', 'auto'):
        return load_custom_dataset(dataset_cfg)
    else:
        raise ValueError(f"Unknown dataset type: {dtype}")