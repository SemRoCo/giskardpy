import traceback
from collections import OrderedDict
from multiprocessing import Lock
from threading import Thread
import rospy
from py_trees import Behaviour, Blackboard, Status


class NewPluginBase(object):
    def __init__(self):
        self.god_map = Blackboard().god_map

    def setup(self):
        pass

    def initialize(self):
        pass

    def stop(self):
        pass

    def update(self):
        return Status.RUNNING

    def get_god_map(self):
        """
        :rtype: giskardpy.god_map.GodMap
        """
        return self.god_map

class GiskardBehavior(Behaviour):
    def __init__(self, name):
        self.god_map = Blackboard().god_map
        super(GiskardBehavior, self).__init__(name)

    def get_god_map(self):
        """
        :rtype: giskardpy.god_map.GodMap
        """
        return self.god_map

    def raise_to_blackboard(self, exception):
        Blackboard().set('exception', exception)

class PluginBehavior(GiskardBehavior):

    def __init__(self, name, sleep=.5):
        self._plugins = OrderedDict()
        self.set_status(Status.INVALID)
        self.status_lock = Lock()
        self.sleep = sleep
        super(PluginBehavior, self).__init__(name)

    def add_plugin(self, name, plugin):
        """Registers a plugin with the process manager. The name needs to be unique."""
        if name in self._plugins:
            raise KeyError(u'A plugin with name "{}" already exists.'.format(name))
        self._plugins[name] = plugin

    def setup(self, timeout):
        self.start_plugins()
        return super(PluginBehavior, self).setup(timeout)

    def start_plugins(self):
        for plugin in self._plugins.values():
            plugin.setup()

    def initialise(self):
        with self.status_lock:
            self.set_status(Status.RUNNING)
        self.update_thread = Thread(target=self.loop_over_plugins)
        self.update_thread.start()
        super(PluginBehavior, self).initialise()

    def init_plugins(self):
        for plugin in self._plugins.values():
            plugin.initialize()

    def is_running(self):
        return self.my_status == Status.RUNNING

    def terminate(self, new_status):
        with self.status_lock:
            self.set_status(Status.FAILURE)
        self.update_thread.join()
        self.stop_plugins()
        super(PluginBehavior, self).terminate(new_status)

    def stop_plugins(self):
        for plugin_name, plugin in self._plugins.items():
            plugin.stop()

    def update(self):
        with self.status_lock:
            if not self.update_thread.is_alive():
                return Status.SUCCESS
            return self.my_status

    def set_status(self, new_state):
        self.my_status = new_state

    def loop_over_plugins(self):
        try:
            self.init_plugins()
            while self.is_running() and not rospy.is_shutdown():
                for plugin_name, plugin in self._plugins.items():
                    with self.status_lock:
                        if not self.is_running():
                            return
                        status = plugin.update()
                        self.set_status(status)
                        assert self.my_status is not None, u'{} did not return a status'.format(plugin_name)
                        if not self.is_running():
                            return
                    rospy.sleep(self.sleep)
        except Exception as e:
            traceback.print_exc()
            # TODO make 'exception' string a parameter somewhere
            Blackboard().set('exception', e)
