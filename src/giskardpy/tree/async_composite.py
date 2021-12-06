import traceback
from collections import OrderedDict
from threading import RLock, Thread

import numpy as np
import rospy
from py_trees import Status, Blackboard

from giskardpy import identifier
from giskardpy.tree.plugin import GiskardBehavior
from giskardpy.utils import logging


class PluginBehavior(GiskardBehavior):

    def __init__(self, name, hz=False, sleep=.5):
        super(PluginBehavior, self).__init__(name)
        self._plugins = OrderedDict()
        self.set_status(Status.INVALID)
        self.status_lock = RLock()
        self.sleep = sleep
        self.looped_once = False
        if hz is not None:
            hz = 1/self.god_map.get_data(identifier.sample_period)
            self.sleeper = rospy.Rate(hz)
        else:
            self.sleeper = None

    def get_plugins(self):
        return self._plugins

    def add_plugin(self, plugin):
        """
        Registers a plugin with the process manager. The name needs to be unique.
        :param plugin: Behaviour
        :return:
        """
        name = plugin.name
        if name in self._plugins:
            raise KeyError('A plugin with name "{}" already exists.'.format(name))
        with self.status_lock:
            self._plugins[name] = plugin

    def remove_plugin(self, plugin_name):
        with self.status_lock:
            del self._plugins[plugin_name]

    def setup(self, timeout):
        self.start_plugins()
        return super(PluginBehavior, self).setup(timeout)

    def start_plugins(self):
        for plugin in self._plugins.values():
            plugin.setup(10.0)

    def initialise(self):
        self.looped_once = False
        with self.status_lock:
            self.set_status(Status.RUNNING)
        self.sleeps = []
        self.update_thread = Thread(target=self.loop_over_plugins)
        self.update_thread.start()
        super(PluginBehavior, self).initialise()

    def init_plugins(self):
        for plugin in self._plugins.values():
            plugin.initialise()

    def is_running(self):
        return self.my_status == Status.RUNNING

    def terminate(self, new_status):
        with self.status_lock:
            self.set_status(Status.FAILURE)
        try:
            self.update_thread.join()
            data = np.array(self.sleeps)
            print(np.average(data))
            print(np.std(data))
        except Exception as e:
            # FIXME sometimes terminate gets called without init being called
            # happens when a previous plugin fails
            logging.logwarn('terminate was called before init')
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
            # self.init_plugins()
            while self.is_running() and not rospy.is_shutdown():
                for plugin_name, plugin in self._plugins.items():
                    with self.status_lock:
                        if not self.is_running():
                            return
                        for node in plugin.tick():
                            status = node.status
                        if status is not None:
                            self.set_status(status)
                        assert self.my_status is not None, '{} did not return a status'.format(plugin_name)
                        if not self.is_running():
                            return
                self.looped_once = True
                if self.sleeper:
                    a = rospy.get_rostime()
                    self.sleeper.sleep()
                    self.sleeps.append((rospy.get_rostime() - a).to_sec())
        except Exception as e:
            traceback.print_exc()
            # TODO make 'exception' string a parameter somewhere
            Blackboard().set('exception', e)
