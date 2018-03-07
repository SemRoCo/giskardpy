from collections import OrderedDict

from giskardpy.databus import DataBus
from giskardpy.exceptions import NameConflictException


class ProcessManager(object):
    def __init__(self):
        self._plugins = OrderedDict()
        self._data_bus = DataBus()

    def register_plugin(self, name, plugin):
        if name in self._plugins:
            raise NameConflictException('A plugin with name "{}" already exists.'.format(name))
        self._plugins[name] = plugin

    def start(self):
        for plugin in self._plugins.values():
            plugin.start(self._data_bus)

    def stop(self):
        for plugin in self._plugins.values():
            plugin.stop()

    def update(self):
        for plugin in self._plugins.values():
            plugin.update()
            for identifier, value in plugin.get_readings().items():
                self._data_bus.set_data(identifier, value)



