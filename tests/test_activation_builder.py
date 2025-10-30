import torch
import torch.nn as nn

from evo_core.formula_generator import FormulaTree, FormulaNode, NodeType
from evo_core.activation_builder import compile_formula_to_torch


def build_sin_x_tree():
    # sin(x)
    x = FormulaNode(NodeType.VARIABLE, "x")
    sin_node = FormulaNode(NodeType.UNARY_OP, "sin", left=x)
    return FormulaTree(sin_node)


def build_x_plus_one_tree():
    # (x + 1.0)
    x = FormulaNode(NodeType.VARIABLE, "x")
    one = FormulaNode(NodeType.CONSTANT, 1.0)
    plus = FormulaNode(NodeType.BINARY_OP, "+", left=x, right=one)
    return FormulaTree(plus)


def test_compile_sin_x_produces_correct_values():
    tree = build_sin_x_tree()
    activation_fn, params = compile_formula_to_torch(tree)
    assert isinstance(params, list) and len(params) == 0, "sin(x) has no trainable constants by default"

    x = torch.linspace(-3.0, 3.0, steps=101).unsqueeze(1)
    with torch.no_grad():
        y = activation_fn(x)
    assert y.shape == x.shape, "Output shape must match input shape"
    assert torch.allclose(y, torch.sin(x), atol=1e-5), "sin(x) compilation should match torch.sin"


def test_compile_x_plus_one_with_and_without_trainable_constants():
    tree = build_x_plus_one_tree()

    # Without trainable constants
    activation_fn, params = compile_formula_to_torch(tree, trainable_constants=False)
    assert len(params) == 0, "No trainable constants expected"
    x = torch.linspace(-2.0, 2.0, steps=21).unsqueeze(1)
    with torch.no_grad():
        y = activation_fn(x)
    assert torch.allclose(y, x + 1.0, atol=1e-6), "x + 1.0 must evaluate correctly"

    # With trainable constants
    activation_fn_tc, params_tc = compile_formula_to_torch(tree, trainable_constants=True)
    # Trigger evaluation so parameters list is populated
    x = torch.linspace(-2.0, 2.0, steps=21).unsqueeze(1)
    with torch.no_grad():
        y_tc = activation_fn_tc(x)
    assert len(params_tc) == 1, "One trainable parameter expected for constant 1.0"
    assert isinstance(params_tc[0], nn.Parameter), "Parameter must be nn.Parameter"
    pval = params_tc[0].item()
    assert torch.allclose(y_tc, x + pval, atol=1e-6), "Output must reflect trainable constant value"