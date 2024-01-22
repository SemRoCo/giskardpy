import threading
import time
from typing import Callable

import rospy
from rospy import Rate
from rospy.timer import TimerEvent


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
