from copy import deepcopy
from typing import Optional

from line_profiler import profile

import giskardpy_ros.ros1.msg_converter as msg_converter
import numpy as np
import rospy
from sensor_msgs.msg import JointState
from py_trees import Status

from giskard_msgs.msg import ExecutionState
from giskardpy.data_types.data_types import LifeCycleState
from giskardpy.god_map import god_map
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time
from giskardpy_ros.tree.blackboard_utils import catch_and_raise_to_blackboard, GiskardBlackboard


def giskard_state_to_execution_state() -> ExecutionState:
    task_filter = np.array([task.plot for task in god_map.motion_graph_manager.task_state.nodes])
    monitor_filter = np.array([monitor.plot for monitor in god_map.motion_graph_manager.monitor_state.nodes])
    msg = ExecutionState()
    msg.header.stamp = rospy.Time.now()
    msg.goal_id = GiskardBlackboard().move_action_server.goal_id
    msg.tasks = [msg_converter.motion_graph_node_to_ros_msg(t) for t in god_map.motion_graph_manager.task_state.nodes if t.plot]
    msg.monitors = [msg_converter.motion_graph_node_to_ros_msg(m) for m in god_map.motion_graph_manager.monitor_state.nodes if m.plot]
    try:
        msg.task_state = god_map.motion_graph_manager.task_state_history[-1][1][0][task_filter].tolist()
        msg.task_life_cycle_state = god_map.motion_graph_manager.task_state_history[-1][1][1][task_filter].tolist()
    except Exception as e:  # state not initialized yet
        msg.task_state = [0] * len(msg.tasks)
        msg.task_life_cycle_state = [LifeCycleState.not_started] * len(msg.tasks)
    try:
        msg.monitor_state = god_map.motion_graph_manager.monitor_state_history[-1][1][0][monitor_filter].tolist()
        msg.monitor_life_cycle_state = god_map.motion_graph_manager.monitor_state_history[-1][1][1][monitor_filter].tolist()
    except Exception as e:  # state not initialized yet
        msg.monitor_state = [0] * len(msg.monitors)
        msg.monitor_life_cycle_state = [LifeCycleState.not_started] * len(msg.monitors)
    return msg


def did_state_change() -> bool:
    if len(god_map.motion_graph_manager.task_state_history) == 0:
        return False
    if len(god_map.motion_graph_manager.task_state_history) == 1:
        return True
    last_task_state = god_map.motion_graph_manager.task_state_history[-2][1][0]
    task_state = god_map.motion_graph_manager.task_state_history[-1][1][0]
    if np.any(last_task_state != task_state):
        return True
    last_task_state = god_map.motion_graph_manager.task_state_history[-2][1][1]
    task_state = god_map.motion_graph_manager.task_state_history[-1][1][1]
    if np.any(last_task_state != task_state):
        return True
    last_monitor_state = god_map.motion_graph_manager.monitor_state_history[-2][1][0]
    monitor_state = god_map.motion_graph_manager.monitor_state_history[-1][1][0]
    if np.any(last_monitor_state != monitor_state):
        return True
    last_monitor_state = god_map.motion_graph_manager.monitor_state_history[-2][1][1]
    monitor_state = god_map.motion_graph_manager.monitor_state_history[-1][1][1]
    if np.any(last_monitor_state != monitor_state):
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
        self.pub = rospy.Publisher(self.cmd_topic, ExecutionState, queue_size=10, latch=True)

    @record_time
    @profile
    def update(self):
        if did_state_change():
            msg = giskard_state_to_execution_state()
            self.pub.publish(msg)
        return Status.SUCCESS
