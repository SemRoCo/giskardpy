from collections import OrderedDict
from copy import copy
from time import sleep

import rospy

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
        while self.update() and not rospy.is_shutdown() :
            # TODO make sure this can be properly killed
            sleep(1)

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
                for n, p in self._plugins.items():
                    parallel_universe.register_plugin(n, p.get_replacement_parallel_universe())
                parallel_universe.start_loop()
                parallel_universe.stop()
                plugin.post_mortem_analysis(parallel_universe.get_god_map())
            for identifier, value in plugin.get_readings().items():
                self._god_map.set_data(identifier, value)
            if plugin.end_parallel_universe():
                print('destroying parallel universe')
                return False
        return True



