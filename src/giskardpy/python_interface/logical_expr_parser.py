import ast
from typing import Dict

import dill


def get_context(f) -> Dict[str, str]:
    if f.__closure__:
        return {value: cell.cell_contents for cell, value in zip(f.__closure__, f.__code__.co_freevars)}
    else:
        return {name: value for name, value in globals().items() if isinstance(value, str)}


def logical_lambda_to_str(f):
    context = get_context(f)
    # Serialize and deserialize the lambda function using dill
    logical_expression_serialized = dill.dumps(f)
    logical_expression_deserialized = dill.loads(logical_expression_serialized)
    # Use ast to parse the deserialized function
    source_code = dill.source.getsource(logical_expression_deserialized)
    source_code = 'lambda' + source_code.split('lambda')[1]
    while source_code:
        try:
            expr_ast = ast.parse(source_code)
            break
        except Exception as e:
            source_code = source_code[:-1]

    # Extract the logical expression (body of the lambda) from the AST
    logical_expr_ast = next(node for node in ast.walk(expr_ast) if isinstance(node, ast.Lambda)).body
    result = ast_to_string(logical_expr_ast)
    for name, value in context.items():
        result = result.replace(name, value)
    return result


def ast_to_string(node):
    if isinstance(node, ast.BoolOp):
        op = ' and ' if isinstance(node.op, ast.And) else ' or '
        return '(' + op.join(ast_to_string(value) for value in node.values) + ')'
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return 'not (' + ast_to_string(node.operand) + ')'
    elif isinstance(node, ast.Name):
        return node.id
    else:
        raise ValueError("Unsupported expression type")
