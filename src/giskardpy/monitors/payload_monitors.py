import abc
from abc import ABC
from threading import Lock
from typing import List, Optional, Dict, Tuple

import numpy as np
import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState

from giskard_msgs.msg import MoveResult, GiskardError
from giskardpy.exceptions import GiskardException, MonitorInitalizationException
from giskardpy.monitors.monitors import Monitor
from giskardpy.god_map import god_map
from giskardpy.utils import logging
import giskardpy.casadi_wrapper as cas


class PayloadMonitor(Monitor, ABC):
    state: bool
    run_call_in_thread: bool

    def __init__(self, *,
                 run_call_in_thread: bool,
                 name: Optional[str] = None,
                 stay_true: bool = True,
                 start_condition: cas.Expression = cas.TrueSymbol):
        """
        A monitor which executes its __call__ function when start_condition becomes True.
        Subclass this and implement __init__ and __call__. The __call__ method should change self.state to True when
        it's done.
        :param run_call_in_thread: if True, calls __call__ in a separate thread. Use for expensive operations
        """
        self.state = False
        self.run_call_in_thread = run_call_in_thread
        super().__init__(name=name, start_condition=start_condition, stay_true=stay_true)

    def get_state(self) -> bool:
        return self.state

    @abc.abstractmethod
    def __call__(self):
        pass


class WorldUpdatePayloadMonitor(PayloadMonitor):
    world_lock = Lock()

    def __init__(self, *,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol):
        super().__init__(name=name, start_condition=start_condition, run_call_in_thread=True)

    @abc.abstractmethod
    def apply_world_update(self):
        pass

    def __call__(self):
        with WorldUpdatePayloadMonitor.world_lock:
            self.apply_world_update()
        self.state = True


class EndMotion(PayloadMonitor):
    def __init__(self,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,):
        super().__init__(name=name, start_condition=start_condition, run_call_in_thread=False)

    def __call__(self):
        self.state = True

    def get_state(self) -> bool:
        return self.state


class CancelMotion(PayloadMonitor):
    def __init__(self,
                 error_message: str,
                 error_code: int = GiskardError.ERROR,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,):
        super().__init__(name=name, start_condition=start_condition, run_call_in_thread=False)
        self.error_message = error_message
        self.error_code = error_code

    @profile
    def __call__(self):
        self.state = True
        raise GiskardException.from_error_code(error_code=self.error_code, error_message=self.error_message)

    def get_state(self) -> bool:
        return self.state


class SetMaxTrajectoryLength(CancelMotion):
    new_length: float

    def __init__(self,
                 new_length: Optional[float] = None,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,):
        if not (start_condition == cas.TrueSymbol).evaluate():
            raise MonitorInitalizationException(f'Cannot set start_condition for {SetMaxTrajectoryLength.__name__}')
        if new_length is None:
            self.new_length = god_map.qp_controller_config.max_trajectory_length
        else:
            self.new_length = new_length
        error_message = f'Trajectory longer than {self.new_length}'
        super().__init__(name=name,
                         start_condition=start_condition,
                         error_message=error_message,
                         error_code=GiskardError.MAX_TRAJECTORY_LENGTH)

    @profile
    def __call__(self):
        if god_map.time > self.new_length:
            return super().__call__()


class Print(PayloadMonitor):
    def __init__(self,
                 message: str,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol):
        self.message = message
        super().__init__(name=name, start_condition=start_condition, run_call_in_thread=False)

    def __call__(self):
        logging.loginfo(self.message)
        self.state = True


class Sleep(PayloadMonitor):
    def __init__(self,
                 seconds: float,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol):
        self.seconds = seconds
        super().__init__(name=name, start_condition=start_condition, run_call_in_thread=True)

    def __call__(self):
        rospy.sleep(self.seconds)
        self.state = True


class UpdateParentLinkOfGroup(WorldUpdatePayloadMonitor):
    def __init__(self,
                 group_name: str,
                 parent_link: str,
                 parent_link_group: Optional[str] = '',
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol):
        if not god_map.is_standalone():
            raise MonitorInitalizationException(f'This monitor can only be used in standalone mode.')
        self.group_name = group_name
        self.new_parent_link = god_map.world.search_for_link_name(parent_link, parent_link_group)
        super().__init__(name=name, start_condition=start_condition)

    def apply_world_update(self):
        god_map.world.move_group(group_name=self.group_name,
                                 new_parent_link_name=self.new_parent_link)
        rospy.sleep(2)


class CollisionMatrixUpdater(PayloadMonitor):
    collision_matrix: Dict[Tuple[str, str], float]

    def __init__(self,
                 new_collision_matrix: Dict[Tuple[str, str], float],
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol):
        super().__init__(name=name, start_condition=start_condition, run_call_in_thread=False)
        self.collision_matrix = new_collision_matrix

    @profile
    def __call__(self):
        god_map.collision_scene.set_collision_matrix(self.collision_matrix)
        god_map.collision_scene.reset_cache()
        self.state = True


class PayloadAlternator(PayloadMonitor):

    def __init__(self,
                 mod: int = 2,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol):
        super().__init__(name=name, stay_true=False, start_condition=start_condition, run_call_in_thread=False)
        self.mod = mod

    def __call__(self):
        self.state = np.floor(god_map.time) % self.mod == 0


class ManipulabilityMonitor(PayloadMonitor):
    def __init__(self,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol):
        super().__init__(name=name, stay_true=False, start_condition=start_condition, run_call_in_thread=False)
        self.first_edge = False
        self.first_down = False

    def __call__(self):
        m = god_map.qp_controller.manipulability_indexes[0]
        m_old = god_map.qp_controller.manipulability_indexes[1]
        if m == 0:
            m = 0.0001
        percentual_diff = 1 - min(m_old / m, 1)
        if percentual_diff < 0.01 and not self.first_edge:
            self.first_edge = True
        if percentual_diff > 0.01 and self.first_edge:
            self.first_down = True
        if percentual_diff < 0.01 and self.first_down:
            self.state = True


class CloseGripper(PayloadMonitor):
    def __init__(self,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 effort: int = -180,
                 pub_topic='hsrb4s/hand_motor_joint_velocity_controller/command',
                 joint_state_topic='hsrb4s/joint_states',
                 velocity_threshold=0.1,
                 effort_threshold=-1,
                 joint_name='hand_motor_joint',
                 as_open=False,
                 motion_goal_name=None
                 ):
        super().__init__(name=name, start_condition=start_condition, run_call_in_thread=False, stay_true=False)
        self.pub = rospy.Publisher(pub_topic, Float64, queue_size=1)
        self.effort = 0
        self.velocity_threshold = velocity_threshold
        rospy.Subscriber(joint_state_topic, JointState, self.callback)
        self.joint_name = joint_name
        self.effort_threshold = effort_threshold
        self.effort_cmd = effort
        self.as_open = as_open
        self.msg = Float64()
        self.msg.data = effort
        self.motion_goal_name = motion_goal_name
        self.msg_e = Float64()
        self.msg_e.data = 0
        if self.as_open:
            self.cmd = 'putdown'
        else:
            self.cmd = 'pickup'
        self.stopped = False

    def __call__(self, *args, **kwargs):
        # read motion goal state from the godmap to publish zero once
        is_active = god_map.motion_goal_manager.motion_goals[self.motion_goal_name].all_commands[self.cmd]
        if is_active:
            self.pub.publish(self.msg)
            self.stopped = False
        elif self.as_open and not is_active and not self.stopped:
            self.pub.publish(self.msg_e)
            self.stopped = True

        if not self.as_open and self.effort < self.effort_threshold:
            self.state = True
        elif self.as_open and self.effort > self.effort_threshold:
            self.state = True
        else:
            self.state = False

    def callback(self, joints: JointState):
        for name, effort, velocity in zip(joints.name, joints.effort, joints.velocity):
            if self.joint_name in name:
                if abs(velocity) < self.velocity_threshold:
                    self.effort = effort
                else:
                    self.effort = 0
        # self.pub.publish(self.msg)