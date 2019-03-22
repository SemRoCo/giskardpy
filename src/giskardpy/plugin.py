import traceback
from collections import OrderedDict
from multiprocessing import Lock
from threading import Thread
import rospy
from py_trees import Behaviour, Blackboard, Status

from giskardpy.identifier import world_identifier, robot_identifier


class PluginBase(object):
    def __init__(self):
        self.god_map = Blackboard().god_map

    def setup(self):
        pass

    def initialize(self):
        pass

    def stop(self):
        pass

    def update(self):
        """
        :return: Status.Success, if the job of the behavior plugin is finished
                 Status.Failure, if something went wrong the the behavior is stopped
                 Status.Running, if the behavior should only be killed in emergencies
                 None, does not change the current status of the behavior
        """
        return None

    def get_god_map(self):
        """
        :rtype: giskardpy.god_map.GodMap
        """
        return self.god_map

    def get_world(self):
        """
        :rtype: giskardpy.world.World
        """
        return self.get_god_map().safe_get_data(world_identifier)

    def get_robot(self):
        """
        :rtype: giskardpy.symengine_robot.Robot
        """
        return self.get_god_map().safe_get_data(robot_identifier)

class GiskardBehavior(Behaviour):
    def __init__(self, name):
        self.god_map = Blackboard().god_map
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
        return self.get_god_map().safe_get_data(world_identifier)

    def get_robot(self):
        """
        :rtype: giskardpy.symengine_robot.Robot
        """
        return self.get_god_map().safe_get_data(robot_identifier)

    def raise_to_blackboard(self, exception):
        Blackboard().set('exception', exception)

class PluginBehavior(GiskardBehavior):

    def __init__(self, name, sleep=.5):
        self._plugins = OrderedDict()
        self.set_status(Status.INVALID)
        self.status_lock = Lock()
        self.sleep = sleep
        self.looped_once = False
        super(PluginBehavior, self).__init__(name)

    def get_plugins(self):
        return self._plugins

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
        self.looped_once = False
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
        try:
            self.update_thread.join()
        except Exception as e:
            # FIXME sometimes terminate gets called without init being called
            rospy.logwarn(u'terminate was called before init')
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

class SuccessPlugin(PluginBase):
    def update(self):
        return Status.SUCCESS
