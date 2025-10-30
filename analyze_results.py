#!/usr/bin/env python
"""
Script for analyzing and visualizing results from evolutionary experiments.
"""

import os
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path


def load_results(results_file):
    """Load results from a JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def plot_formula(formula_str, output_path):
    """Plot the activation function curve via parser + compiler."""
    from evo_core.utils import visualize_formula_from_string
    try:
        fig = visualize_formula_from_string(formula_str)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Formula plot saved to {output_path}")
    except Exception as e:
        print(f"Error plotting formula using parser/compiler: {e}")
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Formula: {formula_str}\n(Visualization not available)",
                 ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title("Activation Function Visualization")
        plt.savefig(output_path)
        plt.close()
        print(f"Placeholder plot saved to {output_path}")


def plot_fitness_history(history, output_path):
    """Plot fitness curves (avg/min/max and best overall)."""
    if not history:
        print("No history found in results; skipping fitness plot.")
        return
    gens = [h['generation'] for h in history]
    avg = [h['avg_fitness'] for h in history]
    min_f = [h['min_fitness'] for h in history]
    max_f = [h['max_fitness'] for h in history]
    best_overall = [h['best_fitness_overall'] for h in history]

    plt.figure(figsize=(10, 6))
    plt.plot(gens, avg, label='Avg fitness', linewidth=2)
    plt.plot(gens, min_f, label='Min fitness', linestyle='--', alpha=0.7)
    plt.plot(gens, max_f, label='Max fitness', linestyle='--', alpha=0.7)
    plt.plot(gens, best_overall, label='Best overall', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Over Generations')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path)
    print(f"Fitness plot saved to {output_path}")


def plot_diversity_history(history, output_path):
    """Plot diversity curve (unique formulas per generation)."""
    if not history:
        print("No history found in results; skipping diversity plot.")
        return
    gens = [h['generation'] for h in history]
    div = [h.get('diversity', 0) for h in history]

    plt.figure(figsize=(10, 6))
    plt.plot(gens, div, label='Diversity', color='purple', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Unique formulas')
    plt.title('Population Diversity Over Generations')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path)
    print(f"Diversity plot saved to {output_path}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze EvoActiv Results")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to results JSON file")
    parser.add_argument("--output", type=str, default="results",
                        help="Directory to save analysis outputs")
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.results)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Extract best formula
    best_formula = results.get("best_formula", "Unknown")
    best_fitness = results.get("best_fitness", 0)
    history = results.get("history", [])
    
    print(f"Best formula: {best_formula}")
    print(f"Best fitness: {best_fitness}")
    print(f"Generations in history: {len(history)}")
    
    # Plot the formula
    plot_path = os.path.join(args.output, "formula_plot.png")
    plot_formula(best_formula, plot_path)

    # Plot fitness history
    fitness_plot_path = os.path.join(args.output, "fitness_history.png")
    plot_fitness_history(history, fitness_plot_path)

    # Plot diversity history
    diversity_plot_path = os.path.join(args.output, "diversity_history.png")
    plot_diversity_history(history, diversity_plot_path)
    
    # Generate summary report
    report_path = os.path.join(args.output, "analysis_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"EvoActiv Analysis Report\n")
        f.write(f"======================\n\n")
        f.write(f"Best Formula: {best_formula}\n")
        f.write(f"Best Fitness: {best_fitness}\n\n")
        f.write(f"Generations: {len(history)}\n")
        if history:
            f.write(f"Last Generation Stats:\n")
            last = history[-1]
            f.write(f"  Avg: {last['avg_fitness']}, Min: {last['min_fitness']}, Max: {last['max_fitness']}\n")
            f.write(f"  Best overall: {last['best_fitness_overall']}, Diversity: {last.get('diversity', 0)}\n\n")
        f.write(f"Configuration:\n")
        for key, value in results.get("config", {}).items():
            f.write(f"  {key}: {value}\n")
    
    print(f"Analysis report saved to {report_path}")


if __name__ == "__main__":
    main()