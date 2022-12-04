import traceback
from collections import OrderedDict
from threading import RLock, Thread
from time import time

import rospy
from py_trees import Status

from giskardpy import identifier
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.utils import raise_to_blackboard


class AsyncBehavior(GiskardBehavior):

    def __init__(self, name, hz=None):
        super().__init__(name)
        self._children = OrderedDict()
        self.set_status(Status.INVALID)
        self.status_lock = RLock()
        self.looped_once = False
        if hz is not None:
            hz = 1/self.god_map.get_data(identifier.sample_period)
            self.sleeper = rospy.Rate(hz)
        else:
            self.sleeper = None

    def get_children(self):
        return self._children

    def add_child(self, child):
        """
        Registers a plugin with the process manager. The name needs to be unique.
        :param child: Behaviour
        :return:
        """
        name = child.name
        if name in self._children:
            raise KeyError(f'A plugin with name \'{name}\' already exists.')
        with self.status_lock:
            self._children[name] = child

    def remove_child(self, child_name):
        with self.status_lock:
            del self._children[child_name]

    def setup(self, timeout):
        self.start_children()
        return super().setup(timeout)

    def start_children(self):
        for child in self._children.values():
            child.setup(10.0)

    def initialise(self):
        self.looped_once = False
        with self.status_lock:
            self.set_status(Status.RUNNING)
        self.update_thread = Thread(target=self.loop_over_plugins)
        self.update_thread.start()
        super().initialise()

    def init_plugins(self):
        for plugin in self._children.values():
            plugin.initialise()
            # plugin.status = Status.RUNNING

    def is_running(self):
        return self.my_status == Status.RUNNING

    def terminate(self, new_status):
        with self.status_lock:
            self.set_status(Status.FAILURE)
        try:
            self.update_thread.join()
        except Exception as e:
            # happens when a previous plugin fails
            # logging.logwarn('terminate was called before init')
            pass
        self.stop_children()
        super().terminate(new_status)

    def stop_children(self):
        for plugin_name, plugin in self._children.items():
            plugin.stop()

    def update(self):
        with self.status_lock:
            if not self.update_thread.is_alive():
                return Status.SUCCESS
            return self.my_status

    def set_status(self, new_state):
        self.my_status = new_state

    @profile
    def loop_over_plugins(self):
        try:
            # self.init_plugins()
            self.get_blackboard().runtime = time()
            while self.is_running() and not rospy.is_shutdown():
                for plugin_name, child in self._children.items():
                    with self.status_lock:
                        if not self.is_running():
                            return
                        for node in child.tick():
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
        except Exception as e:
            traceback.print_exc()
            raise_to_blackboard(e)
