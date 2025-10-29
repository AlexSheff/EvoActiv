"""
Evolution module for EvoActiv.
Handles population management, selection, mutation, and crossover of formulas.
"""

import random
import numpy as np
from typing import List, Tuple, Callable, Dict, Any

from evo_core.formula_generator import FormulaTree
from evo_core.train_evaluator import evaluate_formula
from evo_core.utils import log_generation_stats


class EvolutionEngine:
    """Main class for managing the evolutionary process of activation functions."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the evolution engine with configuration parameters.
        
        Args:
            config: Dictionary containing evolution parameters
        """
        self.population_size = config.get('population_size', 100)
        self.mutation_rate = config.get('mutation_rate', 0.2)
        self.crossover_rate = config.get('crossover_rate', 0.7)
        self.elitism_count = config.get('elitism_count', 5)
        self.tournament_size = config.get('tournament_size', 3)
        self.max_formula_depth = config.get('max_formula_depth', 5)
        self.population = []
        self.generation = 0
        self.best_individual = None
        self.best_fitness = float('-inf')
        
    def initialize_population(self) -> None:
        """Generate initial random population of formula trees."""
        self.population = []
        for _ in range(self.population_size):
            formula = FormulaTree.random_tree(max_depth=self.max_formula_depth)
            self.population.append(formula)
        self.generation = 0
        
    def evaluate_population(self, dataset, evaluation_config: Dict[str, Any]) -> List[float]:
        """
        Evaluate fitness for all individuals in the population.
        
        Args:
            dataset: Dataset to evaluate formulas on
            evaluation_config: Configuration for evaluation process
            
        Returns:
            List of fitness scores for each individual
        """
        fitness_scores = []
        
        for formula in self.population:
            fitness = evaluate_formula(formula, dataset, evaluation_config)
            fitness_scores.append(fitness)
            
            # Update best individual if needed
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_individual = formula.clone()
                
        return fitness_scores
    
    def selection(self, fitness_scores: List[float]) -> List[FormulaTree]:
        """
        Select individuals for reproduction using tournament selection.
        
        Args:
            fitness_scores: List of fitness values for current population
            
        Returns:
            Selected individuals for next generation
        """
        selected = []
        
        # Elitism: keep best individuals
        if self.elitism_count > 0:
            elite_indices = np.argsort(fitness_scores)[-self.elitism_count:]
            for idx in elite_indices:
                selected.append(self.population[idx].clone())
        
        # Tournament selection for the rest
        while len(selected) < self.population_size:
            tournament_indices = random.sample(range(len(self.population)), self.tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(self.population[winner_idx].clone())
            
        return selected
    
    def crossover(self, parent1: FormulaTree, parent2: FormulaTree) -> Tuple[FormulaTree, FormulaTree]:
        """
        Perform crossover between two parent formulas.
        
        Args:
            parent1: First parent formula tree
            parent2: Second parent formula tree
            
        Returns:
            Two offspring formula trees
        """
        if random.random() < self.crossover_rate:
            child1, child2 = parent1.crossover(parent2)
            return child1, child2
        else:
            return parent1.clone(), parent2.clone()
    
    def mutate(self, individual: FormulaTree) -> FormulaTree:
        """
        Apply mutation to an individual formula.
        
        Args:
            individual: Formula tree to mutate
            
        Returns:
            Mutated formula tree
        """
        if random.random() < self.mutation_rate:
            return individual.mutate(max_depth=self.max_formula_depth)
        return individual
    
    def evolve_generation(self, fitness_scores: List[float]) -> None:
        """
        Create a new generation through selection, crossover, and mutation.
        
        Args:
            fitness_scores: Fitness scores of current population
        """
        # Select individuals for reproduction
        selected = self.selection(fitness_scores)
        
        # Create new population through crossover and mutation
        new_population = []
        
        # Add elites directly
        new_population.extend(selected[:self.elitism_count])
        
        # Process the rest through crossover and mutation
        i = self.elitism_count
        while i < len(selected):
            if i + 1 < len(selected):
                child1, child2 = self.crossover(selected[i], selected[i+1])
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])
                i += 2
            else:
                child = self.mutate(selected[i])
                new_population.append(child)
                i += 1
        
        # Ensure population size remains constant
        self.population = new_population[:self.population_size]
        self.generation += 1
        
    def run_evolution(self, dataset, evaluation_config: Dict[str, Any], 
                     num_generations: int, callback: Callable = None) -> FormulaTree:
        """
        Run the complete evolutionary process.
        
        Args:
            dataset: Dataset to evaluate formulas on
            evaluation_config: Configuration for evaluation
            num_generations: Number of generations to evolve
            callback: Optional callback function called after each generation
            
        Returns:
            Best formula found during evolution
        """
        self.initialize_population()
        
        for gen in range(num_generations):
            # Evaluate current population
            fitness_scores = self.evaluate_population(dataset, evaluation_config)
            
            # Log statistics
            log_generation_stats(gen, self.population, fitness_scores, self.best_individual, self.best_fitness)
            
            # Call user callback if provided
            if callback:
                callback(gen, self.population, fitness_scores, self.best_individual)

            # Optional fine-tuning of top-K formulas
            fine_tune_cfg = evaluation_config.get('fine_tuning', {})
            if fine_tune_cfg.get('enable', False):
                top_k = int(fine_tune_cfg.get('top_k', 3))
                extra_epochs = int(fine_tune_cfg.get('extra_epochs', 3))
                # Indices of top-K by fitness
                top_indices = list(np.argsort(fitness_scores)[-top_k:])
                # Prepare config for fine-tuning with more epochs and formula string return
                ft_config = dict(evaluation_config)
                ft_config['epochs'] = int(evaluation_config.get('epochs', 5)) + extra_epochs
                ft_config['return_formula_string'] = True
                # Collect improvements
                improvements = []
                for idx in top_indices:
                    formula = self.population[idx]
                    tuned = evaluate_formula(formula, dataset, ft_config)
                    if isinstance(tuned, tuple):
                        tuned_fitness, tuned_formula_str = tuned
                    else:
                        tuned_fitness, tuned_formula_str = tuned, str(formula)
                    improvements.append({
                        'generation': gen,
                        'formula': str(formula),
                        'tuned_formula': tuned_formula_str,
                        'original_fitness': float(fitness_scores[idx]),
                        'tuned_fitness': float(tuned_fitness),
                    })
                # Attach improvements to engine for external saving
                if not hasattr(self, 'fine_tuning_log'):
                    self.fine_tuning_log = []
                self.fine_tuning_log.append({
                    'generation': gen,
                    'top_k': top_k,
                    'extra_epochs': extra_epochs,
                    'improvements': improvements
                })
            
            # Stop if we've reached the final generation
            if gen == num_generations - 1:
                break
                
            # Create next generation
            self.evolve_generation(fitness_scores)
            
        return self.best_individual