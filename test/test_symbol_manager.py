import unittest

from giskardpy.exceptions import GiskardException
from giskardpy.symbol_manager import symbol_manager


class TestSymbolMananger(unittest.TestCase):
    def test_logic_str_to_expr1(self):
        expr = '(not (d) and (e2 or f f))'
        symbol_manager.get_symbol('d')
        symbol_manager.get_symbol('e2')
        symbol_manager.get_symbol('f f')
        symbol_manager.logic_str_to_expr(expr)

    def test_logic_str_to_expr_fail(self):
        expr = '(not (d) and (e or '
        symbol_manager.get_symbol('d')
        symbol_manager.get_symbol('e')
        symbol_manager.get_symbol('f')
        try:
            symbol_manager.logic_str_to_expr(expr)
            raise Exception('expected exception')
        except SyntaxError:
            pass

    def test_logic_str_to_expr_unknown_symbol(self):
        expr = '(not (d) and (e or q))'
        symbol_manager.get_symbol('d')
        symbol_manager.get_symbol('e')
        symbol_manager.get_symbol('f')
        try:
            symbol_manager.logic_str_to_expr(expr)
            raise Exception('expected exception')
        except GiskardException as e:
            pass

