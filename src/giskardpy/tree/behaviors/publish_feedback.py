from copy import deepcopy
from typing import Optional

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from py_trees import Status

from giskard_msgs.msg import ExecutionState
from giskardpy.data_types import TaskState
from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import catch_and_raise_to_blackboard, record_time


def giskard_state_to_execution_state() -> ExecutionState:
    monitor_filter = np.array([monitor.plot for monitor in god_map.monitor_manager.monitors])
    task_filter = np.array([task.plot for task in god_map.motion_goal_manager.tasks.values()])
    msg = ExecutionState()
    msg.header.stamp = rospy.Time.now()
    msg.goal_id = god_map.move_action_server.goal_id
    msg.monitors = [m.to_ros_msg() for m in god_map.monitor_manager.monitors if m.plot]
    msg.tasks = [t.to_ros_msg() for t in god_map.motion_goal_manager.tasks.values() if t.plot]
    try:
        msg.monitor_state = god_map.monitor_manager.state_history[-1][1][0][monitor_filter].tolist()
        msg.monitor_life_cycle_state = god_map.monitor_manager.state_history[-1][1][1][monitor_filter].tolist()
        msg.task_state = god_map.motion_goal_manager.task_state[task_filter].tolist()
    except Exception as e:  # state not initialized yet
        msg.monitor_state = [0] * len(msg.monitors)
        msg.monitor_life_cycle_state = [TaskState.not_started] * len(msg.monitors)
        msg.task_state = [TaskState.not_started] * len(msg.tasks)
    return msg


def did_state_change() -> bool:
    if len(god_map.monitor_manager.state_history) == 0:
        return False
    if len(god_map.monitor_manager.state_history) == 1:
        return True
    # monitor state
    if np.any(god_map.monitor_manager.state_history[-1][1][0] != god_map.monitor_manager.state_history[-2][1][0]):
        return True
    # lifecycle state
    if np.any(god_map.monitor_manager.state_history[-1][1][1] != god_map.monitor_manager.state_history[-2][1][1]):
        return True
    # lifecycle state
    if np.any(god_map.motion_goal_manager.state_history[-1][1] != god_map.motion_goal_manager.state_history[-2][1]):
        return True
    return False


class PublishFeedback(GiskardBehavior):
    @profile
    def __init__(self, name: Optional[str] = None, topic_name: Optional[str] = None):
        if name is None:
            name = self.__class__.__name__
        if topic_name is None:
            topic_name = '~state'
        super().__init__(name)
        self.cmd_topic = topic_name
        self.pub = rospy.Publisher(self.cmd_topic, ExecutionState, queue_size=10)

    @record_time
    @profile
    def update(self):
        if did_state_change():
            msg = giskard_state_to_execution_state()
            self.pub.publish(msg)
        return Status.SUCCESS
