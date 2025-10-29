"""
Train evaluator module for EvoActiv.
Handles training and evaluation of neural networks with evolved activation functions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

from evo_core.activation_builder import ActivationModule
from evo_core.formula_generator import FormulaTree


class SimpleNet(nn.Module):
    """Simple neural network for testing activation functions."""
    
    def __init__(self, activation_fn, hidden_layers: List[int] = [128, 64], dropout_rate: float = 0.2):
        """
        Initialize the network.
        
        Args:
            activation_fn: Activation function to use
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # Input layer (MNIST: 28x28 = 784)
        input_size = 784
        
        # Hidden layers
        for hidden_size in hidden_layers:
            self.layers.append(nn.Linear(input_size, hidden_size))
            self.layers.append(activation_fn)
            self.layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size
        
        # Output layer (MNIST: 10 classes)
        self.layers.append(nn.Linear(input_size, 10))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Flatten the input
        x = x.view(x.size(0), -1)
        
        # Pass through all layers
        for layer in self.layers:
            x = layer(x)
            
        return x


def create_and_evaluate_model(formula: FormulaTree, 
                             dataset: Tuple[DataLoader, DataLoader],
                             config: Dict[str, Any]) -> Any:
    """
    Create a model with the given activation function and evaluate it.
    
    Args:
        formula: Formula tree representing the activation function
        dataset: Tuple of (train_loader, val_loader)
        config: Configuration dictionary
        
    Returns:
        Fitness score (validation accuracy)
    """
    # Extract dataset
    train_loader, val_loader = dataset
    
    # Formula optimization mode: joint | finetune_only | none
    fo_cfg = config.get('formula_optimization', {})
    mode = fo_cfg.get('mode', 'joint')
    # Create activation function with trainable constants if enabled
    # Prefer formula_optimization.enable, fallback to operators.trainable_constants
    trainable_constants = fo_cfg.get('enable', config.get('operators', {}).get('trainable_constants', False))
    if mode == 'none':
        trainable_constants = False
    activation_module = ActivationModule(formula, trainable_constants=trainable_constants)

    # Set device early for safety checks and model placement
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Safety check: validate activation on a small random batch to avoid NaN/Inf
    try:
        test_x = torch.randn(16, 784, device=device, requires_grad=True)
        # Pass through one linear layer + activation to simulate typical usage
        test_layer = nn.Linear(784, 32).to(device)
        with torch.enable_grad():
            y = activation_module(test_layer(test_x))
            # Sum and backprop to check gradients
            y.sum().backward()
        if not torch.isfinite(y).all() or not torch.isfinite(test_x.grad).all():
            print(f"Activation produced non-finite values or gradients: {activation_module}")
            return 0.0 if not config.get('return_formula_string', False) else (0.0, str(formula))
    except Exception as e:
        print(f"Activation safety check failed for {formula}: {e}")
        return 0.0 if not config.get('return_formula_string', False) else (0.0, str(formula))
    
    # Create model
    model = SimpleNet(
        activation_fn=activation_module,
        hidden_layers=config.get('hidden_layers', [128, 64]),
        dropout_rate=config.get('dropout_rate', 0.2)
    )
    
    # Place model on device
    model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer setup based on mode
    base_lr = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 0.0001)
    formula_opt_config = fo_cfg
    formula_lr = formula_opt_config.get('learning_rate', base_lr)
    # Optional multiplier relative to base lr
    lr_mult = formula_opt_config.get('parameter_lr_mult', 1.0)
    formula_lr = formula_lr if 'learning_rate' in formula_opt_config else base_lr * lr_mult

    if mode == 'joint':
        if trainable_constants and activation_module.get_params():
            param_groups = [
                {'params': model.parameters(), 'lr': base_lr},
                {'params': activation_module.get_params(), 'lr': formula_lr}
            ]
            optimizer = optim.Adam(param_groups, weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    elif mode == 'finetune_only':
        # Freeze model, optimize only activation params
        for p in model.parameters():
            p.requires_grad = False
        if trainable_constants and activation_module.get_params():
            optimizer = optim.Adam(activation_module.get_params(), lr=formula_lr, weight_decay=0.0)
        else:
            # No trainable params: fallback to standard
            optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    else:  # none
        optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    
    # Training loop
    epochs = config.get('epochs', 5)
    patience = config.get('early_stopping_patience', 3)
    
    best_val_acc = 0.0
    patience_counter = 0
    
    # Store original formula for logging
    original_formula = str(formula)
    
    # Track parameter changes if trainable
    param_history = []
    if trainable_constants and activation_module.get_params():
        param_history.append({
            'epoch': 0,
            'params': [p.item() for p in activation_module.get_params()]
        })
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                # Calculate accuracy
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        # Calculate validation accuracy
        val_acc = correct / total
        
        # Log parameter values after each epoch if trainable
        if trainable_constants and activation_module.get_params():
            param_history.append({
                'epoch': epoch + 1,
                'params': [p.item() for p in activation_module.get_params()],
                'val_acc': val_acc
            })
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Log final formula with trained parameters
    final_formula_str = None
    if trainable_constants and activation_module.get_params():
        final_formula_str = activation_module.get_current_formula()
        print(f"Original formula: {original_formula}")
        print(f"Final formula: {activation_module}")
        print(f"Parameter history: {param_history}")
    
    # Return enriched info if requested
    if config.get('return_formula_string', False) and final_formula_str is not None:
        return best_val_acc, final_formula_str
    
    return best_val_acc


def evaluate_formula(formula: FormulaTree, 
                    dataset: Tuple[DataLoader, DataLoader],
                    config: Dict[str, Any]) -> float:
    """
    Evaluate a formula by training a model with it as the activation function.
    
    Args:
        formula: Formula tree to evaluate
        dataset: Dataset to train and evaluate on
        config: Configuration for model training
        
    Returns:
        Fitness score (validation accuracy)
    """
    try:
        return create_and_evaluate_model(formula, dataset, config)
    except Exception as e:
        print(f"Error evaluating formula {formula}: {e}")
        return 0.0  # Return minimum fitness for failed formulas