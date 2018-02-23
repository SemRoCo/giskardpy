from collections import OrderedDict

from giskardpy.databus import DataBus
from giskardpy.exceptions import NameConflictException
from giskardpy.plugin import InputPlugin, OutputPlugin


class ProcessManager(object):
    def __init__(self):
        #TODO there are plugins that are both in and output
        self._input_plugins = OrderedDict()
        self._output_plugins = OrderedDict()
        self._data_bus = DataBus()

    def register_plugin(self, name, plugin):
        if isinstance(plugin, InputPlugin):
            self._register_input_plugin(name, plugin)
        elif isinstance(plugin, OutputPlugin):
            self._register_output_plugin(name, plugin)

    def _register_input_plugin(self, name, plugin):
        if name in self._input_plugins:
            raise NameConflictException('A plugin with name "{}" already exists.'.format(name))
        self._input_plugins[name] = plugin

    def _register_output_plugin(self, name, plugin):
        if name in self._output_plugins:
            raise NameConflictException('A plugin with name "{}" already exists.'.format(name))
        self._output_plugins[name] = plugin

    def start(self):
        for plugin in self._input_plugins.values():
            plugin.start()
        for plugin in self._output_plugins.values():
            plugin.start()

    def stop(self):
        for plugin in self._input_plugins.values():
            plugin.stop()
        for plugin in self._output_plugins.values():
            plugin.stop()

    def update(self):
        for plugin in self._input_plugins.values():
            for identifier, value in plugin.get_readings().items():
                self._data_bus.set_data(identifier, value)
        for plugin in self._output_plugins.values():
            plugin.update(self._data_bus)



