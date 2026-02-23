from __future__ import annotations

from typing import Callable, Dict
import ast
import math


_ALLOWED_FUNCS: Dict[str, Callable[..., float]] = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "log": math.log,
    "sqrt": math.sqrt,
    "abs": abs,
    "pow": pow,
    "min": min,
    "max": max,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "floor": math.floor,
    "ceil": math.ceil,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
}

_ALLOWED_NAMES = {"x", "pi", "e"}


class _ExprValidator(ast.NodeVisitor):
    def visit_Expression(self, node: ast.Expression) -> None:
        self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        self.visit(node.operand)

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise ValueError("Only whitelisted functions are allowed")
        for arg in node.args:
            self.visit(arg)
        if node.keywords:
            raise ValueError("Keyword arguments are not allowed")

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in _ALLOWED_NAMES:
            raise ValueError("Unknown name: {}".format(node.id))

    def visit_Constant(self, node: ast.Constant) -> None:
        if not isinstance(node.value, (int, float)):
            raise ValueError("Only numeric constants are allowed")

    def generic_visit(self, node: ast.AST) -> None:
        allowed = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Call,
            ast.Name,
            ast.Constant,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.USub,
            ast.UAdd,
        )
        if not isinstance(node, allowed):
            raise ValueError("Unsupported expression element: {}".format(type(node).__name__))
        super().generic_visit(node)


def compile_expression(expr: str) -> Callable[[float], float]:
    expr = expr.strip()
    if not expr:
        raise ValueError("Expression is empty")
    expr = expr.replace("^", "**")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid expression syntax") from exc
    _ExprValidator().visit(tree)
    code = compile(tree, "<expr>", "eval")
    safe_env = {"__builtins__": {}}
    safe_env.update(_ALLOWED_FUNCS)
    safe_env.update({"pi": math.pi, "e": math.e})

    def _eval(x: float) -> float:
        return float(eval(code, safe_env, {"x": float(x)}))

    return _eval


def load_expression(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        expr = f.read().strip()
    if not expr:
        raise ValueError("Expression file is empty: {}".format(path))
    return expr
