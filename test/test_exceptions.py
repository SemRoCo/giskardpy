import unittest

from giskard_msgs.msg import GiskardError
from giskardpy.exceptions import GiskardException

error_codes = {value: name for name, value in vars(GiskardError).items() if isinstance(value, (int, float))}


class TestExceptions(unittest.TestCase):
    def test_if_exception_for_all_error_code_exists(self):
        for value, code_name in error_codes.items():
            if value != GiskardError.SUCCESS:
                assert value in GiskardException._error_code_map, f'No exception defined for {code_name}'
