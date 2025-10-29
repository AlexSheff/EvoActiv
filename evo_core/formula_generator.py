"""
Formula generator module for EvoActiv.
Handles creation and manipulation of formula syntax trees.
"""

import random
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from enum import Enum


class NodeType(Enum):
    """Enum for node types in formula tree."""
    VARIABLE = "variable"
    CONSTANT = "constant"
    UNARY_OP = "unary_op"
    BINARY_OP = "binary_op"


class FormulaNode:
    """Node in the formula syntax tree."""
    
    def __init__(self, node_type: NodeType, value: Any = None, 
                 left: Optional['FormulaNode'] = None, 
                 right: Optional['FormulaNode'] = None):
        """
        Initialize a formula node.
        
        Args:
            node_type: Type of the node (variable, constant, unary_op, binary_op)
            value: Value of the node (operator name, variable name, or constant value)
            left: Left child node
            right: Right child node
        """
        self.node_type = node_type
        self.value = value
        self.left = left
        self.right = right
        
    def clone(self) -> 'FormulaNode':
        """Create a deep copy of this node and its children."""
        left_clone = self.left.clone() if self.left else None
        right_clone = self.right.clone() if self.right else None
        return FormulaNode(self.node_type, self.value, left_clone, right_clone)
    
    def __str__(self) -> str:
        """Convert node to string representation."""
        if self.node_type == NodeType.VARIABLE:
            return str(self.value)
        elif self.node_type == NodeType.CONSTANT:
            return str(self.value)
        elif self.node_type == NodeType.UNARY_OP:
            return f"{self.value}({self.left})"
        elif self.node_type == NodeType.BINARY_OP:
            return f"({self.left} {self.value} {self.right})"
        return ""


class FormulaTree:
    """Represents a complete formula as a syntax tree."""
    
    # Available operations for formula generation
    UNARY_OPS = ["sin", "cos", "exp", "log", "tanh", "sigmoid", "relu"]
    BINARY_OPS = ["+", "-", "*", "/", "**"]
    
    def __init__(self, root: FormulaNode = None):
        """
        Initialize a formula tree.
        
        Args:
            root: Root node of the formula tree
        """
        self.root = root
        
    def clone(self) -> 'FormulaTree':
        """Create a deep copy of this formula tree."""
        if self.root is None:
            return FormulaTree()
        return FormulaTree(self.root.clone())
    
    def __str__(self) -> str:
        """Convert formula tree to string representation."""
        if self.root is None:
            return "Empty Formula"
        return str(self.root)
    
    @staticmethod
    def random_tree(max_depth: int = 3, 
                    variable_name: str = "x", 
                    p_leaf: float = 0.3,
                    p_constant: float = 0.5,
                    depth: int = 0) -> 'FormulaTree':
        """
        Generate a random formula tree.
        
        Args:
            max_depth: Maximum depth of the tree
            variable_name: Name of the input variable
            p_leaf: Probability of generating a leaf node at non-max depth
            p_constant: Probability of a leaf being a constant vs a variable
            depth: Current depth (used in recursion)
            
        Returns:
            A randomly generated formula tree
        """
        # At max depth or randomly at lower depths, create a leaf node
        if depth >= max_depth or (depth > 0 and random.random() < p_leaf):
            if random.random() < p_constant:
                # Generate random constant between -3 and 3
                value = random.uniform(-3, 3)
                node = FormulaNode(NodeType.CONSTANT, value)
            else:
                # Use the variable
                node = FormulaNode(NodeType.VARIABLE, variable_name)
            return FormulaTree(node)
        
        # Decide between unary and binary operation
        if random.random() < 0.4:  # 40% chance for unary op
            op = random.choice(FormulaTree.UNARY_OPS)
            left_subtree = FormulaTree.random_tree(max_depth, variable_name, p_leaf, p_constant, depth + 1)
            node = FormulaNode(NodeType.UNARY_OP, op, left_subtree.root)
        else:  # 60% chance for binary op
            op = random.choice(FormulaTree.BINARY_OPS)
            left_subtree = FormulaTree.random_tree(max_depth, variable_name, p_leaf, p_constant, depth + 1)
            right_subtree = FormulaTree.random_tree(max_depth, variable_name, p_leaf, p_constant, depth + 1)
            node = FormulaNode(NodeType.BINARY_OP, op, left_subtree.root, right_subtree.root)
            
        return FormulaTree(node)
    
    def get_nodes_list(self) -> List[Tuple[FormulaNode, Optional[FormulaNode], int]]:
        """
        Get a list of all nodes with their parent and child index.
        
        Returns:
            List of tuples (node, parent, child_index)
            where child_index is 0 for left child, 1 for right child
        """
        nodes = []
        
        def traverse(node, parent=None, child_idx=None):
            if node is None:
                return
            nodes.append((node, parent, child_idx))
            traverse(node.left, node, 0)
            traverse(node.right, node, 1)
            
        traverse(self.root)
        return nodes
    
    def mutate(self, max_depth: int = 3) -> 'FormulaTree':
        """
        Create a mutated copy of this formula tree.
        
        Args:
            max_depth: Maximum depth for any new subtrees
            
        Returns:
            A new mutated formula tree
        """
        # Clone the tree first
        mutated = self.clone()
        
        # Get all nodes with their parents
        nodes = mutated.get_nodes_list()
        if not nodes:
            return mutated
        
        # Select a random node to mutate
        node, parent, child_idx = random.choice(nodes)
        
        # Generate a replacement subtree
        replacement = FormulaTree.random_tree(max_depth=max_depth // 2).root
        
        # Replace the selected node
        if parent is None:
            # Replacing the root
            mutated.root = replacement
        else:
            # Replacing a child node
            if child_idx == 0:
                parent.left = replacement
            else:
                parent.right = replacement
                
        return mutated
    
    def crossover(self, other: 'FormulaTree') -> Tuple['FormulaTree', 'FormulaTree']:
        """
        Perform crossover with another formula tree.
        
        Args:
            other: Another formula tree to crossover with
            
        Returns:
            Two new formula trees resulting from crossover
        """
        # Clone both trees
        child1 = self.clone()
        child2 = other.clone()
        
        # Get nodes from both trees
        nodes1 = child1.get_nodes_list()
        nodes2 = child2.get_nodes_list()
        
        # Check if both trees have nodes
        if not nodes1 or not nodes2:
            return child1, child2
        
        # Select random nodes from each tree
        node1, parent1, child_idx1 = random.choice(nodes1)
        node2, parent2, child_idx2 = random.choice(nodes2)
        
        # Swap the selected subtrees
        if parent1 is None:
            # node1 is the root
            if parent2 is None:
                # Both are roots, swap the entire trees
                child1.root, child2.root = child2.root, child1.root
            else:
                # node2 is not root
                temp = child1.root
                child1.root = node2
                if child_idx2 == 0:
                    parent2.left = temp
                else:
                    parent2.right = temp
        else:
            # node1 is not the root
            if parent2 is None:
                # node2 is the root
                temp = node1
                if child_idx1 == 0:
                    parent1.left = child2.root
                else:
                    parent1.right = child2.root
                child2.root = temp
            else:
                # Neither is root, straightforward swap
                if child_idx1 == 0:
                    temp = parent1.left
                    parent1.left = node2
                else:
                    temp = parent1.right
                    parent1.right = node2
                    
                if child_idx2 == 0:
                    parent2.left = temp
                else:
                    parent2.right = temp
                    
        return child1, child2