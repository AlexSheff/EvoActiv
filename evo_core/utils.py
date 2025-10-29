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

from evo_core.formula_generator import FormulaTree


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
    activation_fn = compile_formula_to_torch(formula)
    
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