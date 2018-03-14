from collections import OrderedDict

import rospy
from copy import deepcopy, copy

from giskardpy.god_map import GodMap
from giskardpy.exceptions import NameConflictException


class ProcessManager(object):
    def __init__(self, initial_state=None):
        self._plugins = OrderedDict()
        self._god_map = GodMap() if initial_state is None else copy(initial_state)

    def register_plugin(self, name, plugin):
        if name in self._plugins:
            raise NameConflictException('A plugin with name "{}" already exists.'.format(name))
        self._plugins[name] = plugin

    def start_loop(self):
        for plugin in self._plugins.values():
            plugin.start(self._god_map)
        while not rospy.is_shutdown() and self.update():
            # rospy.sleep(0.5)
            pass

    def stop(self):
        for plugin in self._plugins.values():
            plugin.stop()

    def get_god_map(self):
        return self._god_map

    def update(self):
        for plugin_name, plugin in self._plugins.items():
            plugin.update()
            if plugin.create_parallel_universe():
                print('creating new parallel universe')
                parallel_universe = ProcessManager(initial_state=self._god_map)
                for name, plugin in self._plugins.items():
                    parallel_universe.register_plugin(name, plugin.copy())
                parallel_universe.start_loop()
                parallel_universe.stop()
                for plugin_name2, plugin2 in self._plugins.items():
                    # TODO might be enough to call this on plugin instead of all of them
                    plugin2.post_mortem_analysis(parallel_universe.get_god_map())
            for identifier, value in plugin.get_readings().items():
                self._god_map.set_data(identifier, value)
            if plugin.end_parallel_universe():
                print('destroying parallel universe')
                return False
        return True



