from __future__ import division

import numbers

from giskardpy.exceptions import GiskardException
from giskardpy.goals.goal import Goal


class UpdateGodMap(Goal):

    def __init__(self, updates, **kwargs):
        """
        Modifies the core data structure of giskard, only use it if you know what you are doing
        """
        super().__init__(**kwargs)
        self.update_god_map([], updates)

    def update_god_map(self, identifier, updates):
        if not isinstance(updates, dict):
            raise GiskardException('{} used incorrectly, {} not a dict or number'.format(str(self), updates))
        for member, value in updates.items():
            next_identifier = identifier + [member]
            if isinstance(value, numbers.Number) and \
                    isinstance(self.god_map.get_data(next_identifier), numbers.Number):
                self.god_map.set_data(next_identifier, value)
            else:
                self.update_god_map(next_identifier, value)
