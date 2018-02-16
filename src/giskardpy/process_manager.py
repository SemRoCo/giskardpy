from giskardpy.exceptions import NameConflictException


class ProcessManager(object):
    def __init__(self):
        self._plugins = {}

    def register_plugin(self, name, plugin):
        if name in self._plugins:
            raise NameConflictException('A plugin with name "{}" already exists.'.format(name))
        # TODO: verify plugin type
        self._plugins[name] = plugin

    def start(self):
        for plugin in self._plugins:
            plugin.start()

    def stop(self):
        for plugin in self._plugins:
            plugin.stop()

    def update(self):
        for plugin in self._plugins:
            plugin.update()