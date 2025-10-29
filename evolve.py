#!/usr/bin/env python
"""
Main script for running evolutionary activation function search.
"""

import os
import argparse
import yaml
import torch
import json
from datetime import datetime

from evo_core.evolution import EvolutionEngine
from evo_core.train_evaluator import create_and_evaluate_model
from datasets.dataset_loader import load_dataset_from_config


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_results(results, config, output_dir):
    """Save experiment results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"results_{timestamp}.json")
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {result_file}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="EvoActiv: Evolutionary Activation Function Search")
    parser.add_argument("--config", type=str, default="config/default.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--output", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--dataset", type=str, choices=["mnist", "custom"], 
                        help="Dataset to use (overrides config)")
    parser.add_argument("--generations", type=int, 
                        help="Number of generations (overrides config)")
    parser.add_argument("--population_size", type=int, 
                        help="Population size (overrides config)")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.dataset:
        config['dataset']['type'] = args.dataset
    if args.generations:
        config['evolution']['num_generations'] = args.generations
    if args.population_size:
        config['evolution']['population_size'] = args.population_size
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    train_loader, val_loader = load_dataset_from_config(config['dataset'])
    
    # Initialize evolution engine
    print("Initializing evolution engine...")
    evolution_engine = EvolutionEngine(config['evolution'])
    
    # Run evolution
    print(f"Starting evolution for {config['evolution']['num_generations']} generations...")
    # Merge model training config with formula optimization for evaluator
    eval_config = {**config.get('model', {}),
                   'formula_optimization': config.get('formula_optimization', {}),
                   'fine_tuning': config.get('fine_tuning', {})}

    # Collect per-generation history
    history: list[dict] = []
    
    def on_generation(gen_idx, population, fitness_scores, best_individual):
        # Compute diversity as count of unique formulas
        unique_formulas = len(set(str(f) for f in population))
        history.append({
            'generation': gen_idx,
            'avg_fitness': float(sum(fitness_scores) / len(fitness_scores)),
            'min_fitness': float(min(fitness_scores)),
            'max_fitness': float(max(fitness_scores)),
            'best_fitness_overall': float(evolution_engine.best_fitness),
            'diversity': int(unique_formulas),
        })
    
    best_formula = evolution_engine.run_evolution(
        dataset=(train_loader, val_loader),
        evaluation_config=eval_config,
        num_generations=config['evolution']['num_generations'],
        callback=on_generation
    )
    
    # Save results
    results = {
        "best_formula": str(best_formula),
        "best_fitness": evolution_engine.best_fitness,
        "history": history,
        "fine_tuning_log": getattr(evolution_engine, 'fine_tuning_log', []),
        "config": config,
        "timestamp": datetime.now().isoformat()
    }
    
    # If formula had trainable constants, add original and final formulas
    if hasattr(best_formula, 'get_original_formula') and hasattr(best_formula, 'get_current_formula'):
        results['original_formula'] = best_formula.get_original_formula()
        results['final_formula'] = best_formula.get_current_formula()
    
    save_results(results, config, args.output)
    
    # Also save to formulas.json (appending to existing file)
    formulas_file = os.path.join(args.output, 'formulas.json')
    all_formulas = []
    
    # Load existing formulas if file exists
    if os.path.exists(formulas_file):
        try:
            with open(formulas_file, 'r') as f:
                all_formulas = json.load(f)
        except:
            all_formulas = []
    
    # Add new formula
    all_formulas.append(results)
    
    # Save all formulas
    with open(formulas_file, 'w') as f:
        json.dump(all_formulas, f, indent=4)
    
    print("Evolution completed successfully!")
    print(f"Best formula: {best_formula}")
    print(f"Best fitness: {evolution_engine.best_fitness}")


if __name__ == "__main__":
    main()