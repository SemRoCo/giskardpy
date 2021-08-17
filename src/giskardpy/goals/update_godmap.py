from __future__ import division

import numbers

from giskardpy.exceptions import GiskardException
from giskardpy.goals.goal import Goal


class UpdateGodMap(Goal):

    def __init__(self, updates, **kwargs):
        """
        Modifies the core data structure of giskard, only use it if you know what you are doing
        """
        super(UpdateGodMap, self).__init__(**kwargs)
        self.update_god_map([], updates)

    def update_god_map(self, identifier, updates):
        if not isinstance(updates, dict):
            raise GiskardException(u'{} used incorrectly, {} not a dict or number'.format(str(self), updates))
        for member, value in updates.items():
            next_identifier = identifier + [member]
            if isinstance(value, numbers.Number) and \
                    isinstance(self.get_god_map().get_data(next_identifier), numbers.Number):
                self.get_god_map().set_data(next_identifier, value)
            else:
                self.update_god_map(next_identifier, value)
