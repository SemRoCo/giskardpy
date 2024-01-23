import traceback
from threading import Thread
from time import time
from typing import Optional

import rospy
from py_trees import Status, Composite

from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.ros_timer import Rate
from giskardpy.utils.utils import raise_to_blackboard


class AsyncBehavior(GiskardBehavior, Composite):
    """
    A composite that runs its children in a different thread.
    Status is Running if all children are Running.
    If one child returns either Success or Failure, this behavior will return it as well.
    """

    def __init__(self, name: str, max_hz: Optional[float] = None):
        """
        :param name:
        :param max_hz: The frequency at which this thread is looped will be limited to this value, if possible.
        """
        super().__init__(name)
        self.set_status(Status.INVALID)
        self.looped_once = False
        self.max_hz = max_hz

    def initialise(self) -> None:
        self.looped_once = False
        self.update_thread = Thread(target=self.loop_over_plugins, name=self.name)
        self.update_thread.start()
        super().initialise()

    def is_running(self) -> bool:
        return self.status == Status.RUNNING

    def terminate(self, new_status: Status) -> None:
        self.set_status(Status.FAILURE)
        try:
            self.update_thread.join()
        except Exception as e:
            # happens when a previous plugin fails
            # logging.logwarn('terminate was called before init')
            pass
        self.stop_children()
        super().terminate(new_status)

    def stop_children(self) -> None:
        for child in self.children:
            child.stop()

    def tick(self):
        if self.status == Status.INVALID:
            self.status = Status.RUNNING
            self.initialise()
        yield self

    def set_status(self, new_state: Status) -> None:
        self.status = new_state

    def tip(self):
        return GiskardBehavior.tip(self)

    @profile
    def loop_over_plugins(self) -> None:
        try:
            self.get_blackboard().runtime = time()
            if self.max_hz is not None:
                self.sleeper = Rate(self.max_hz, complain=True)
            else:
                self.sleeper = None
            while self.is_running() and not rospy.is_shutdown():
                for child in self.children:
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
                    self.sleeper.sleep()
        except Exception as e:
            traceback.print_exc()
            raise_to_blackboard(e)
