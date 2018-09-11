import traceback
from collections import OrderedDict
from multiprocessing import Lock
from threading import Thread
from time import time

import rospy
from py_trees import Behaviour, Blackboard, Status
from smach import State

from giskardpy.exceptions import MAX_NWSR_REACHEDException


class PluginBase(object):
    def __init__(self):
        self.started = False

    def start(self, god_map):
        """
        :param god_map:
        :type god_map: giskardpy.god_map.GodMap
        :return:
        """
        self.god_map = god_map
        if not self.started:
            self.start_once()
            self.started = True
        self.initialize()

    def start_once(self):
        """
        This function is called during when the plugin is started, but never more than once. start_once will be called
        before start_always.
        """
        pass

    def initialize(self):
        """
        This function is called everything the plugin gets started. start_once will be called first.
        """
        pass

    def stop(self):
        """
        This function is called when a parallel universe or the whole program gets terminated.
        """
        # TODO 2 stops? 1 for main loop, 1 for parallel shit
        pass

    def update(self):
        """
        This function is called once during every loop of the process manager.
        god_map should only be modified in this function!
        """
        pass

    def create_parallel_universe(self):
        """
        :return: Whether or not a new parallel universe should be created.
        :rtype: bool
        """
        return False

    def end_parallel_universe(self):
        """
        :return: Whether or not the current parallel universe should cease to exist. Every plugin can trigger the end
                    of a universe, even if it did not trigger its creation.
        :rtype: bool
        """
        return False

    def post_mortem_analysis(self, god_map, exception):
        """
        Analyse the the god map and potential cause of death of the parallel universe.
        This function will only be called for the plugin that triggered the creation of a universe.
        :param god_map: the dead gods memory
        :type god_map: GodMap
        :param exception:
        :type exception: Exception
        """
        pass

    def get_replacement(self):
        """
        This function is called on every plugin when a new universe is created. Useful e.g. to replace a plugin that
        monitors the robot state with a simulation.
        DON'T override this function.
        :rtype: PluginBase
        """
        c = self.copy()
        c.started = self.started
        return c

    def copy(self):
        """
        :return: Used by get_replacement, override this function
        :rtype: PluginBase
        """
        raise (NotImplementedError)


class PluginParallelUniverseOnly(PluginBase):
    """
    This Class can be used if you want a plugin to be only active in a parallel universe
    """

    def __init__(self, replacement, call_start=False):
        """
        :param replacement: Plugin that gets only activated in parallel universes
        :type replacement: PluginBase
        :param call_start: Whether this plugin should call the replacements start. If this is False the plugin will get
                            started during the first parallel universe.
        :type call_start: bool
        """
        self.replacement = replacement
        self.call_init = call_start
        super(PluginParallelUniverseOnly, self).__init__()

    def copy(self):
        return self.replacement

    def start(self, god_map):
        if self.call_init:
            self.replacement.start(god_map)

    def get_replacement(self):
        c = self.copy()
        return c


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
        # self.status = new_status
        with self.status_lock:
            self.set_status(Status.FAILURE)
        self.update_thread.join()
        self.stop_plugins()
        super(PluginBehavior, self).terminate(new_status)

    def stop_plugins(self):
        for plugin_name, plugin in self._plugins.items():
            plugin.stop()

    def update(self):
        # print('update wait')
        with self.status_lock:
            # print('update got')
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
                    # print('loop wait {}'.format(plugin_name))
                    with self.status_lock:
                        # print('loop got {}'.format(plugin_name))
                        if not self.is_running():
                            return
                        status = plugin.update()
                        self.set_status(status)
                        assert self.my_status is not None, u'{} did not return a status'.format(plugin_name)
                        if not self.is_running():
                            return
                    rospy.sleep(self.sleep)
                    # print('looped {}'.format(plugin_name))
                    # rospy.sleep(.1)
        except Exception as e:
            traceback.print_exc()
            # TODO make 'exception' string a parameter somewhere
            Blackboard().set('exception', e)
