import traceback
from collections import OrderedDict
from threading import RLock
from threading import Thread

import rospy
from py_trees import Behaviour, Blackboard, Status

from giskardpy.identifier import world, robot
from giskardpy.utils import logging


class GiskardBehavior(Behaviour):
    def __init__(self, name):
        self.god_map = Blackboard().god_map
        self.world = None
        self.robot = None
        super(GiskardBehavior, self).__init__(name)

    def get_god_map(self):
        """
        :rtype: giskardpy.god_map.GodMap
        """
        return self.god_map

    def get_world(self):
        """
        :rtype: giskardpy.world.World
        """
        if not self.world:
            self.world = self.get_god_map().get_data(world)
        return self.world

    def unsafe_get_world(self):
        """
        :rtype: giskardpy.world.World
        """
        if not self.world:
            self.world = self.get_god_map().unsafe_get_data(world)
        return self.world

    def get_robot(self):
        """
        :rtype: giskardpy.robot.Robot
        """
        if not self.robot:
            self.robot = self.get_god_map().get_data(robot)
        return self.robot

    def unsafe_get_robot(self):
        """
        :rtype: giskardpy.robot.Robot
        """
        if not self.robot:
            self.robot = self.get_god_map().unsafe_get_data(robot)
        return self.robot

    def raise_to_blackboard(self, exception):
        Blackboard().set('exception', exception)

    def get_blackboard(self):
        return Blackboard()

    def get_blackboard_exception(self):
        return self.get_blackboard().get('exception')

    def clear_blackboard_exception(self):
        self.get_blackboard().set('exception', None)


class PluginBehavior(GiskardBehavior):

    def __init__(self, name, sleep=.5):
        self._plugins = OrderedDict()
        self.set_status(Status.INVALID)
        self.status_lock = RLock()
        self.sleep = sleep
        self.looped_once = False
        super(PluginBehavior, self).__init__(name)

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
            raise KeyError(u'A plugin with name "{}" already exists.'.format(name))
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
        except Exception as e:
            # FIXME sometimes terminate gets called without init being called
            # happens when a previous plugin fails
            logging.logwarn(u'terminate was called before init')
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
                        assert self.my_status is not None, u'{} did not return a status'.format(plugin_name)
                        if not self.is_running():
                            return
                self.looped_once = True
        except Exception as e:
            traceback.print_exc()
            # TODO make 'exception' string a parameter somewhere
            Blackboard().set('exception', e)


class SuccessPlugin(GiskardBehavior):
    def update(self):
        return Status.SUCCESS
