import sympy as sp

from giskardpy.qp_problem_builder import SoftConstraint


def minimize_array(list_of_symbols_names, goal_suffix='g', weight_suffix='w'):
    """
    Creates a list of soft constraints which minimize the distance for each symbol from the input to a goal.
    :param list_of_symbols_names:
    :param goal_suffix:
    :param weight_suffix:
    :return:
    """
    goal_symbols = []
    weight_symbols = []
    soft_constraints = []
    for i, symbol_name in enumerate(list_of_symbols_names):
        goal_name = '{}_{}'.format(symbol_name, goal_suffix)
        goal = sp.Symbol(goal_name)
        goal_symbols.append(goal_name)

        weight_name = '{}_{}'.format(goal_name, weight_suffix)
        weight_symbols.append(weight_name)

        symbol = sp.Symbol(symbol_name)
        soft_constraints.append(SoftConstraint(lower=goal - symbol, upper=goal - symbol,
                                               weight=sp.Symbol(weight_name), expression=symbol))
    return goal_symbols, weight_symbols, soft_constraints
