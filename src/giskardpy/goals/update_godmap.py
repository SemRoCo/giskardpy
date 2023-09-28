from __future__ import division

import numbers

from giskardpy.exceptions import GiskardException
from giskardpy.goals.goal import Goal
from giskardpy.god_map_interpreter import god_map


class Updategod_map.Goal):

    def __init__(self, updates):
        """
        Modifies the core data structure of giskard, only use it if you know what you are doing
        """
        super().__init__()
        self.update_god_map([], updates)

    def update_god_map(self, identifier, updates):
        if not isinstance(updates, dict):
            raise GiskardException('{} used incorrectly, {} not a dict or number'.format(str(self), updates))
        for member, value in updates.items():
            next_identifier = identifier + [member]
            if isinstance(value, numbers.Number) and \
                    isinstance(god_map.get_data(next_identifier), numbers.Number):
                god_map.set_data(next_identifier, value)
            else:
                self.update_god_map(next_identifier, value)
