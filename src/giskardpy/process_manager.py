from giskardpy.databus import DataBus
from giskardpy.exceptions import NameConflictException
from giskardpy.plugin import InputPlugin


class ProcessManager(object):
    def __init__(self):
        self._input_plugins = {}
        self._data_bus = DataBus()

    def register_plugin(self, name, plugin):
        if isinstance(plugin, InputPlugin):
            self._register_input_plugin(name, plugin)

    def _register_input_plugin(self, name, plugin):
        if name in self._input_plugins:
            raise NameConflictException('A plugin with name "{}" already exists.'.format(name))
        self._input_plugins[name] = plugin

    def start(self):
        for plugin in self._input_plugins:
            plugin.start()

    def stop(self):
        for plugin in self._input_plugins:
            plugin.stop()

    def update(self):
        for plugin in self._input_plugins:
            for identifier, value in plugin.get_readings().items():
                self._data_bus.set_data(identifier, value)



