import numpy as np

import giskardpy.casadi_wrapper as cas

a = cas.Symbol('a')
b = cas.Symbol('b')

expr = a + b
print(expr)
# (a+b)

expr2 = cas.greater(a, b)
print(expr2)
# (b<a)

expr_compiled = expr.compile(parameters=[a, b])
print(expr_compiled(a=1, b=2))
# [3.]

expr2_compiled = expr2.compile(parameters=[a, b])
print(expr2_compiled(a=1, b=2))
# [0.]

print(expr2_compiled.fast_call(np.array([2, 1])))
# [1.]

expr3 = a**2
expr3_gradient = cas.jacobian([expr3], [a])
print(expr3_gradient)
# (a+a)
