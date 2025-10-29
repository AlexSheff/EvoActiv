"""
Activation builder module for EvoActiv.
Compiles formula trees into PyTorch functions.
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
import os
from typing import Dict, Any, Callable, List, Tuple, Optional

from evo_core.formula_generator import FormulaTree, NodeType


# Load configuration
def load_config():
    """Load configuration from default.yaml"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'default.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Get default configuration
CONFIG = load_config()


def compile_formula_to_torch(formula: FormulaTree, trainable_constants: bool = False) -> Tuple[Callable, List[nn.Parameter]]:
    """
    Compile a formula tree into a PyTorch-compatible activation function.
    
    Args:
        formula: The formula tree to compile
        trainable_constants: Whether constants should be trainable parameters
        
    Returns:
        A tuple of (activation_function, parameters_list)
    """
    # List to store all trainable parameters
    params = []
    
    def activation_function(x: torch.Tensor) -> torch.Tensor:
        return evaluate_node(formula.root, x, params, trainable_constants)
    
    return activation_function, params


# Define operation handlers
UNARY_OPS = {
    "sin": lambda x: torch.sin(x),
    "cos": lambda x: torch.cos(x),
    "exp": lambda x: torch.exp(torch.clamp(x, -10, 10)),
    "log": lambda x: torch.log(torch.abs(x) + 1e-6),
    "tanh": lambda x: torch.tanh(x),
    "sigmoid": lambda x: torch.sigmoid(x),
    "relu": lambda x: torch.relu(x)
}

BINARY_OPS = {
    "+": lambda x, y: x + y,
    "-": lambda x, y: x - y,
    "*": lambda x, y: x * y,
    "/": lambda x, y: x / (torch.abs(y) + 1e-6),
    "**": lambda x, y: torch.pow(x, torch.clamp(y, -2, 2))
}

# Get operators from config
def get_operators_from_config():
    """Get operators from configuration"""
    unary_ops = CONFIG.get('operators', {}).get('unary', list(UNARY_OPS.keys()))
    binary_ops = CONFIG.get('operators', {}).get('binary', list(BINARY_OPS.keys()))
    
    # Filter to only include supported operations
    unary_ops = [op for op in unary_ops if op in UNARY_OPS]
    binary_ops = [op for op in binary_ops if op in BINARY_OPS]
    
    return unary_ops, binary_ops

# Get operators
ENABLED_UNARY_OPS, ENABLED_BINARY_OPS = get_operators_from_config()

def evaluate_node(node, x: torch.Tensor, params: List[nn.Parameter] = None, 
                 trainable_constants: bool = False) -> torch.Tensor:
    """
    Recursively evaluate a formula node with PyTorch operations.
    
    Args:
        node: The formula node to evaluate
        x: Input tensor
        params: List to store trainable parameters
        trainable_constants: Whether constants should be trainable parameters
        
    Returns:
        Result of evaluating the node
    """
    if params is None:
        params = []
        
    if node.node_type == NodeType.VARIABLE:
        return x
    
    if node.node_type == NodeType.CONSTANT:
        if trainable_constants:
            # Create a trainable parameter for this constant
            param = nn.Parameter(torch.tensor(float(node.value)))
            # Store the parameter and its original node for later reference
            node.param_index = len(params)
            params.append(param)
            return param * torch.ones_like(x)
        else:
            return torch.ones_like(x) * node.value
    
    if node.node_type == NodeType.UNARY_OP:
        operand = evaluate_node(node.left, x, params, trainable_constants)
        
        # Apply the unary operation
        if node.value in UNARY_OPS:
            return UNARY_OPS[node.value](operand)
        else:
            raise ValueError(f"Unknown or disabled unary operation: {node.value}")
    
    if node.node_type == NodeType.BINARY_OP:
        left = evaluate_node(node.left, x, params, trainable_constants)
        right = evaluate_node(node.right, x, params, trainable_constants)
        
        # Apply the binary operation
        if node.value in BINARY_OPS:
            return BINARY_OPS[node.value](left, right)
        else:
            raise ValueError(f"Unknown or disabled binary operation: {node.value}")
    
    raise ValueError(f"Unknown node type: {node.node_type}")


class ActivationModule(nn.Module):
    """PyTorch module wrapper for evolved activation functions."""
    
    def __init__(self, formula: FormulaTree, trainable_constants: bool = None):
        """
        Initialize activation module with a formula.
        
        Args:
            formula: Formula tree to use as activation function
            trainable_constants: Whether constants should be trainable parameters
        """
        super().__init__()
        self.formula = formula
        self.original_formula_str = str(formula)
        
        # Check if trainable_constants is specified, otherwise use config
        if trainable_constants is None:
            trainable_constants = CONFIG.get('formula_optimization', {}).get('enable', False)
        
        self.trainable_constants = trainable_constants
        
        # Compile the formula and get the parameters
        self.activation_fn, self.params = compile_formula_to_torch(formula, trainable_constants)
        
        # Register parameters with the module
        for i, param in enumerate(self.params):
            self.register_parameter(f'param_{i}', param)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the activation function to input tensor.
        
        Args:
            x: Input tensor
            
        Returns:
            Activated tensor
        """
        return self.activation_fn(x)
    
    def get_params(self) -> List[nn.Parameter]:
        """Return list of trainable parameters."""
        return self.params
    
    def get_original_formula(self) -> str:
        """Return the original formula string before training."""
        return self.original_formula_str
    
    def get_current_formula(self) -> str:
        """Return the current formula with updated parameter values."""
        if not self.trainable_constants or not self.params:
            return str(self.formula)
        
        # Create a deep copy of the formula for parameter substitution
        import copy
        formula_copy = copy.deepcopy(self.formula)
        
        # Function to recursively update constant nodes with trained values
        def update_constants(node):
            if hasattr(node, 'param_index'):
                # Update the constant value with the trained parameter value
                node.value = float(self.params[node.param_index].item())
            
            # Recursively process children
            if hasattr(node, 'left') and node.left is not None:
                update_constants(node.left)
            if hasattr(node, 'right') and node.right is not None:
                update_constants(node.right)
        
        # Update constants in the formula copy
        update_constants(formula_copy.root)
        
        return str(formula_copy)
    
    def __str__(self) -> str:
        """Return string representation of the activation function."""
        return self.get_current_formula()