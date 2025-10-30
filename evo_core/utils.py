"""
Utility functions for EvoActiv.
Provides logging, timing, and visualization utilities.
"""

import time
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
from datetime import datetime

from evo_core.formula_generator import FormulaTree, FormulaNode, NodeType


def log_generation_stats(generation: int, 
                         population: List[FormulaTree], 
                         fitness_scores: List[float],
                         best_individual: FormulaTree,
                         best_fitness: float) -> None:
    """
    Log statistics for the current generation.
    
    Args:
        generation: Current generation number
        population: List of individuals in the population
        fitness_scores: List of fitness scores
        best_individual: Best individual found so far
        best_fitness: Best fitness score found so far
    """
    avg_fitness = np.mean(fitness_scores)
    min_fitness = np.min(fitness_scores)
    max_fitness = np.max(fitness_scores)
    
    print(f"Generation {generation}:")
    print(f"  Average Fitness: {avg_fitness:.4f}")
    print(f"  Min Fitness: {min_fitness:.4f}")
    print(f"  Max Fitness: {max_fitness:.4f}")
    print(f"  Best Fitness Overall: {best_fitness:.4f}")
    print(f"  Best Formula: {best_individual}")
    print("-" * 50)


def save_generation_stats(generation: int,
                         population: List[FormulaTree],
                         fitness_scores: List[float],
                         best_individual: FormulaTree,
                         best_fitness: float,
                         output_dir: str) -> None:
    """
    Save generation statistics to a JSON file.
    
    Args:
        generation: Current generation number
        population: List of individuals in the population
        fitness_scores: List of fitness scores
        best_individual: Best individual found so far
        best_fitness: Best fitness score found so far
        output_dir: Directory to save the statistics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    stats = {
        "generation": generation,
        "timestamp": datetime.now().isoformat(),
        "avg_fitness": float(np.mean(fitness_scores)),
        "min_fitness": float(np.min(fitness_scores)),
        "max_fitness": float(np.max(fitness_scores)),
        "best_fitness_overall": float(best_fitness),
        "best_formula": str(best_individual),
        "top_formulas": [
            {
                "formula": str(population[i]),
                "fitness": float(fitness_scores[i])
            }
            for i in np.argsort(fitness_scores)[-10:]  # Top 10 formulas
        ]
    }
    
    file_path = os.path.join(output_dir, f"generation_{generation}.json")
    with open(file_path, 'w') as f:
        json.dump(stats, f, indent=2)


def plot_fitness_history(fitness_history: List[Dict[str, float]], 
                         output_path: str) -> None:
    """
    Plot the fitness history over generations.
    
    Args:
        fitness_history: List of dictionaries with fitness statistics
        output_path: Path to save the plot
    """
    generations = [stats["generation"] for stats in fitness_history]
    avg_fitness = [stats["avg_fitness"] for stats in fitness_history]
    min_fitness = [stats["min_fitness"] for stats in fitness_history]
    max_fitness = [stats["max_fitness"] for stats in fitness_history]
    best_overall = [stats["best_fitness_overall"] for stats in fitness_history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, avg_fitness, label='Average Fitness')
    plt.plot(generations, min_fitness, label='Min Fitness')
    plt.plot(generations, max_fitness, label='Max Fitness')
    plt.plot(generations, best_overall, label='Best Overall', linestyle='--')
    
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Evolution')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(output_path)
    plt.close()


class Timer:
    """Simple timer for measuring execution time."""
    
    def __init__(self, name: str = "Operation"):
        """
        Initialize the timer.
        
        Args:
            name: Name of the operation being timed
        """
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        """Start the timer when entering a context."""
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Print the elapsed time when exiting the context."""
        elapsed_time = time.time() - self.start_time
        print(f"{self.name} completed in {elapsed_time:.2f} seconds")


def visualize_formula(formula: FormulaTree, output_path: str) -> None:
    """
    Visualize a formula by plotting its output for a range of inputs.
    
    Args:
        formula: Formula tree to visualize
        output_path: Path to save the visualization
    """
    import torch
    from evo_core.activation_builder import compile_formula_to_torch
    
    # Create x values from -5 to 5
    x = np.linspace(-5, 5, 1000)
    x_tensor = torch.FloatTensor(x).reshape(-1, 1)
    
    # Compile the formula to a PyTorch function
    activation_fn, _ = compile_formula_to_torch(formula)
    
    # Evaluate the function
    with torch.no_grad():
        y = activation_fn(x_tensor).numpy()
    
    # Plot the function
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.title(f"Activation Function: {formula}")
    plt.xlabel('x')
    plt.ylabel('f(x)')
    
    # Save the plot
    plt.savefig(output_path)
    plt.close()


def parse_formula_string_to_tree(formula_str: str) -> FormulaTree:
    """
    Parse a formula string into a FormulaTree using sympy as an intermediate representation.

    Supports unary ops: sin, cos, exp, log, tanh, sigmoid, relu
    Supports binary ops: +, -, *, ** (division represented via multiplication with reciprocal if present)
    """
    from sympy import sympify, Symbol, Function

    x = Symbol('x')
    # Register custom functions that are not built-in sympy
    relu = Function('relu')
    sigmoid = Function('sigmoid')

    expr = sympify(formula_str, locals={'x': x, 'relu': relu, 'sigmoid': sigmoid})

    def to_node(e) -> FormulaNode:
        # Variable
        if getattr(e, 'name', None) == 'x' and e.is_Symbol:
            return FormulaNode(NodeType.VARIABLE, 'x')

        # Numeric constant
        if hasattr(e, 'is_Number') and e.is_Number:
            return FormulaNode(NodeType.CONSTANT, float(e))

        # Unary functions
        if hasattr(e, 'func') and hasattr(e, 'args') and len(e.args) == 1:
            fname = str(e.func.__name__).lower()
            if fname in {"sin", "cos", "exp", "log", "tanh", "sigmoid", "relu"}:
                return FormulaNode(NodeType.UNARY_OP, fname, left=to_node(e.args[0]))

        # Binary operations and multi-arg reductions
        # Addition: fold args left-associatively using '+'
        if hasattr(e, 'is_Add') and e.is_Add:
            args = list(e.args)
            n = to_node(args[0])
            for a in args[1:]:
                n = FormulaNode(NodeType.BINARY_OP, '+', left=n, right=to_node(a))
            return n

        # Multiplication: fold using '*'
        if hasattr(e, 'is_Mul') and e.is_Mul:
            args = list(e.args)
            n = to_node(args[0])
            for a in args[1:]:
                n = FormulaNode(NodeType.BINARY_OP, '*', left=n, right=to_node(a))
            return n

        # Power: assume binary '**'
        if hasattr(e, 'is_Pow') and e.is_Pow:
            base, exp = e.as_base_exp()
            return FormulaNode(NodeType.BINARY_OP, '**', left=to_node(base), right=to_node(exp))

        # Fallback: try to interpret as a function call with any name
        if hasattr(e, 'func') and hasattr(e, 'args') and len(e.args) == 1:
            fname = str(e.func.__name__)
            return FormulaNode(NodeType.UNARY_OP, fname, left=to_node(e.args[0]))

        raise ValueError(f"Unsupported expression for parsing: {e}")

    root = to_node(expr)
    return FormulaTree(root)


def visualize_formula_from_string(formula_str: str, trainable_constants: bool = False):
    """
    Build a matplotlib figure visualizing the formula provided as string.
    Parses the string into a FormulaTree, compiles it, evaluates on a grid, and returns the figure.
    """
    import torch
    from evo_core.activation_builder import compile_formula_to_torch

    tree = parse_formula_string_to_tree(formula_str)

    x = np.linspace(-5, 5, 1000)
    x_tensor = torch.FloatTensor(x).reshape(-1, 1)
    activation_fn, _ = compile_formula_to_torch(tree, trainable_constants=trainable_constants)

    with torch.no_grad():
        y = activation_fn(x_tensor).numpy()

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y, linewidth=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.set_title(f"Activation Function: {formula_str}")
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')

    return fig