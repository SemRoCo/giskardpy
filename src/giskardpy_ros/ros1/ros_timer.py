import threading
import time
from typing import Callable

import numpy as np
import rospy
from rospy.timer import TimerEvent
from rospy.timer import sleep

from giskardpy.middleware import get_middleware
from giskardpy_ros.tree.blackboard_utils import GiskardBlackboard
from giskardpy.utils.utils import is_running_in_pytest


class Rate:
    """
    Convenience class for sleeping in a loop at a specified rate
    """

    def __init__(self, hz, reset=False, complain: bool = False):
        """
        Constructor.
        @param hz: hz rate to determine sleeping
        @type  hz: float
        @param reset: if True, timer is reset when rostime moved backward. [default: False]
        @type  reset: bool
        """
        self.print_warning = complain and not is_running_in_pytest()
        self.last_time = rospy.rostime.get_rostime()
        self.sleep_dur = rospy.rostime.Duration(0, int(1e9 / hz))
        self._reset = reset
        self.dt_sum = 0
        self.loop_count = 0

    def _remaining(self, curr_time):
        """
        Calculate the time remaining for rate to sleep.
        @param curr_time: current time
        @type  curr_time: L{Time}
        @return: time remaining
        @rtype: L{Time}
        """
        # detect time jumping backwards
        if self.last_time > curr_time:
            self.last_time = curr_time

        # calculate remaining time
        elapsed = curr_time - self.last_time
        return self.sleep_dur - elapsed

    def remaining(self):
        """
        Return the time remaining for rate to sleep.
        @return: time remaining
        @rtype: L{Time}
        """
        curr_time = rospy.rostime.get_rostime()
        return self._remaining(curr_time)

    def sleep(self):
        """
        Attempt sleep at the specified rate. sleep() takes into
        account the time elapsed since the last successful
        sleep().

        @raise ROSInterruptException: if ROS shutdown occurs before
        sleep completes
        @raise ROSTimeMovedBackwardsException: if ROS time is set
        backwards
        """
        curr_time = rospy.rostime.get_rostime()
        try:
            sleep(self._remaining(curr_time))
        except rospy.exceptions.ROSTimeMovedBackwardsException:
            if not self._reset:
                raise
            self.last_time = rospy.rostime.get_rostime()
            return

        # detect time jumping forwards, as well as loops that are
        # inherently too slow
        elapsed_time = curr_time - self.last_time
        if self.loop_count > 1:
            self.dt_sum += elapsed_time.to_sec()
        self.loop_count += 1
        if elapsed_time > self.sleep_dur * 2:
            self.last_time = curr_time
            if self.print_warning and self.loop_count > 1:
                get_middleware().logwarn(
                    f'Control loop can\'t keep up with {GiskardBlackboard().giskard.qp_controller_config.control_dt} dt. '
                    f'This loop took {elapsed_time.to_sec():.5f}s; average is {self.avg_dt}')
        self.last_time = self.last_time + self.sleep_dur

    @property
    def avg_dt(self):
        if self.loop_count >= 2:
            return self.dt_sum / (self.loop_count - 1)
        return np.nan


class Timer(threading.Thread):
    """
    Convenience class for calling a callback at a specified rate
    """

    def __init__(self, period: rospy.Duration, callback: Callable[[TimerEvent], None], thread_name: str,
                 oneshot: bool = False, reset: bool = False):
        """
        Constructor.
        @param period: desired period between callbacks
        @param callback: callback to be called
        @param oneshot: if True, fire only once, otherwise fire continuously until shutdown is called [default: False]
        @param reset: if True, timer is reset when rostime moved backward. [default: False]
        """
        self.thread_name = thread_name
        super().__init__(name=self.thread_name)
        self._period = period
        self._callback = callback
        self._oneshot = oneshot
        self._reset = reset
        self._shutdown = False
        self.daemon = True
        self.start()

    def shutdown(self):
        """
        Stop firing callbacks.
        """
        self._shutdown = True

    def run(self):
        r = Rate(1.0 / self._period.to_sec(), reset=self._reset)
        current_expected = rospy.rostime.get_rostime() + self._period
        last_expected, last_real, last_duration = None, None, None
        while not rospy.core.is_shutdown() and not self._shutdown:
            try:
                r.sleep()
            except rospy.exceptions.ROSInterruptException as e:
                if rospy.core.is_shutdown():
                    break
                raise
            if self._shutdown:
                break
            current_real = rospy.rostime.get_rostime()
            start = time.time()
            self._callback(TimerEvent(last_expected, last_real, current_expected, current_real, last_duration))
            if self._oneshot:
                break
            last_duration = time.time() - start
            last_expected, last_real = current_expected, current_real
            current_expected += self._period
