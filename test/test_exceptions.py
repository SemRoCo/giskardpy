import unittest

from giskard_msgs.msg import MoveResult
from giskardpy.exceptions import GiskardException

error_codes = {value: name for name, value in vars(MoveResult).items() if isinstance(value, (int, float))}


class TestExceptions(unittest.TestCase):
    def test_if_exception_for_all_error_code_exists(self):
        for value, code_name in error_codes.items():
            if value != MoveResult.SUCCESS:
                assert value in GiskardException._error_code_map, f'No exception defined for {code_name}'
