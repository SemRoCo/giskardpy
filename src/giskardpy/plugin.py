from time import time

import rospy
from py_trees import Behaviour, Blackboard
from smach import State


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
        self.start_always()

    def start_once(self):
        """
        This function is called during when the plugin is started, but never more than once. start_once will be called
        before start_always.
        """
        pass

    def start_always(self):
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


class GiskardState(State):
    Finished = 'next'
    GodMapIOKey = 'god_map'
    def __init__(self, outcomes=[]):
        State.__init__(self,
                       outcomes=[self.Finished]+outcomes,
                       io_keys=[self.GodMapIOKey])

    def get_god_map(self, ud):
        return getattr(ud, self.GodMapIOKey)

class SleepState(GiskardState):
    def execute(self, ud):
        # rospy.sleep(0.1)
        god_map = self.get_god_map(ud)
        if god_map.get_data(['c']) == 10000:
            print(time() - god_map.get_data(['time']))
        god_map.set_data(['c'], god_map.get_data(['c'])+1)
        return self.Finished

class GiskardBehavior(Behaviour):
    def __init__(self, name):
        self.god_map = Blackboard().god_map
        super(GiskardBehavior, self).__init__(name)