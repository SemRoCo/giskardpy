import traceback
from collections import OrderedDict
from threading import RLock, Thread
from time import time

import rospy
from py_trees import Status, Composite

from giskardpy import identifier
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.utils import raise_to_blackboard


class AsyncBehavior(GiskardBehavior, Composite):

    def __init__(self, name, hz=None):
        super().__init__(name)
        self.set_status(Status.INVALID)
        self.status_lock = RLock()
        self.looped_once = False
        if hz is not None:
            hz = 1/self.god_map.get_data(identifier.sample_period)
            self.sleeper = rospy.Rate(hz)
        else:
            self.sleeper = None

    def initialise(self):
        self.looped_once = False
        self.update_thread = Thread(target=self.loop_over_plugins)
        self.update_thread.start()
        super().initialise()

    def is_running(self):
        return self.status == Status.RUNNING

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
        for child in self.children:
            child.stop()

    def tick(self):
        with self.status_lock:
            if self.status == Status.INVALID:
                self.status = Status.RUNNING
                self.initialise()
            yield self

    def set_status(self, new_state):
        self.status = new_state

    def tip(self):
        return GiskardBehavior.tip(self)

    @profile
    def loop_over_plugins(self):
        try:
            self.get_blackboard().runtime = time()
            while self.is_running() and not rospy.is_shutdown():
                for child in self.children:
                    with self.status_lock:
                        if not self.is_running():
                            return
                        for node in child.tick():
                            status = node.status
                        if status is not None:
                            self.set_status(status)
                        assert self.status is not None, f'{child.name} did not return a status'
                        if not self.is_running():
                            return
                self.looped_once = True
                if self.sleeper:
                    a = rospy.get_rostime()
                    self.sleeper.sleep()
        except Exception as e:
            traceback.print_exc()
            raise_to_blackboard(e)
