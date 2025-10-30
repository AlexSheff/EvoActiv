import pytest

from evo_core.formula_generator import FormulaTree, FormulaNode, NodeType


def tree_depth(node):
    if node is None:
        return 0
    left_depth = tree_depth(node.left)
    right_depth = tree_depth(node.right)
    return 1 + max(left_depth, right_depth)


def is_valid_tree(tree: FormulaTree) -> bool:
    return tree is not None and tree.root is not None


def test_random_tree_creation_basic():
    tree = FormulaTree.random_tree(max_depth=3)
    assert is_valid_tree(tree), "Random tree should have a root"
    # Depth counting includes root; generator uses depth starting at 0, so max nodes depth is max_depth + 1
    assert tree_depth(tree.root) <= 4, "Tree depth should not exceed max_depth + 1"
    s = str(tree)
    assert isinstance(s, str) and len(s) > 0, "String representation should be non-empty"


def test_mutate_changes_structure_non_trivially():
    tree = FormulaTree.random_tree(max_depth=4)
    original_str = str(tree)

    # Try multiple times to avoid flakiness if mutation produces identical string
    mutated = None
    for _ in range(5):
        mutated = tree.mutate(max_depth=4)
        if str(mutated) != original_str:
            break

    assert is_valid_tree(mutated), "Mutated tree should be valid"
    assert tree_depth(mutated.root) <= 5, "Mutated tree should respect max_depth + 1"
    assert str(mutated) != original_str, "Mutation should change the tree structure"


def test_crossover_returns_valid_children():
    t1 = FormulaTree.random_tree(max_depth=3)
    t2 = FormulaTree.random_tree(max_depth=3)

    parent_strs = {str(t1), str(t2)}
    child1, child2 = t1.crossover(t2)

    assert is_valid_tree(child1) and is_valid_tree(child2), "Children must be valid trees"
    # Crossover can increase depth; only verify children are valid

    # At least one child should differ from both parents most of the time; try retries to avoid rare identical cases
    attempts = 0
    while attempts < 5 and (str(child1) in parent_strs and str(child2) in parent_strs):
        child1, child2 = t1.crossover(t2)
        attempts += 1

    assert str(child1) not in parent_strs or str(child2) not in parent_strs, "Crossover should produce a different structure in at least one child"