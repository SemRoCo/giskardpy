import giskardpy.casadi_wrapper as cas
from giskardpy.god_map import god_map
from giskardpy.symbol_manager import symbol_manager

god_map.a = 1
god_map.b = 2

a = symbol_manager.get_symbol('god_map.a')
b = symbol_manager.get_symbol('god_map.b')

expr = a + b
print(expr)
# (god_map.a+god_map.b)

expr2 = cas.greater(a, b)
print(expr2)
# (god_map.b<god_map.a)

expr_compiled = expr.compile()
args = symbol_manager.resolve_symbols(expr_compiled.str_params)
print(expr_compiled.fast_call(args))
# [3.]

expr2_compiled = expr2.compile()
args = symbol_manager.resolve_symbols(expr2_compiled.str_params)
print(expr2_compiled.fast_call(args))
# [0.]

god_map.b = -1
args = symbol_manager.resolve_symbols(expr2_compiled.str_params)
print(expr2_compiled.fast_call(args))
# [1.]

expr3 = a ** 2
expr3_gradient = cas.jacobian([expr3], [a])
print(expr3_gradient)
# (god_map.a+god_map.a)
expr3_gradient_compiled = expr3_gradient.compile()
args = symbol_manager.resolve_symbols(expr3_gradient_compiled.str_params)
print(expr3_gradient_compiled.fast_call(args))
# [2.]
