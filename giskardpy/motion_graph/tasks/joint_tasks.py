from typing import Optional, Dict

from giskardpy import casadi_wrapper as cas
from giskardpy.data_types.data_types import Derivatives, PrefixName
from giskardpy.data_types.exceptions import GoalInitalizationException
from giskardpy.god_map import god_map
from giskardpy.model.joints import OneDofJoint
from giskardpy.motion_graph.monitors.joint_monitors import JointGoalReached
from giskardpy.motion_graph.tasks.task import Task, WEIGHT_BELOW_CA


class JointPositionList(Task):
    def __init__(self, *,
                 goal_state: Dict[str, float],
                 group_name: Optional[str] = None,
                 threshold: float = 0.01,
                 weight: float = WEIGHT_BELOW_CA,
                 max_velocity: float = 1,
                 name: Optional[str] = None,
                 plot: bool = True):
        super().__init__(name=name, plot=plot)

        self.current_positions = []
        self.goal_positions = []
        self.velocity_limits = []
        self.joint_names = []
        self.max_velocity = max_velocity
        self.weight = weight
        if len(goal_state) == 0:
            raise GoalInitalizationException(f'Can\'t initialize {self} with no joints.')

        for joint_name, goal_position in goal_state.items():
            joint_name = god_map.world.search_for_joint_name(joint_name, group_name)
            self.joint_names.append(joint_name)

            ll_pos, ul_pos = god_map.world.compute_joint_limits(joint_name, Derivatives.position)
            if ll_pos is not None:
                goal_position = cas.limit(goal_position, ll_pos, ul_pos)

            ll_vel, ul_vel = god_map.world.compute_joint_limits(joint_name, Derivatives.velocity)
            velocity_limit = cas.limit(max_velocity, ll_vel, ul_vel)

            joint: OneDofJoint = god_map.world.joints[joint_name]
            self.current_positions.append(joint.get_symbol(Derivatives.position))
            self.goal_positions.append(goal_position)
            self.velocity_limits.append(velocity_limit)

        for name, current, goal, velocity_limit in zip(self.joint_names, self.current_positions,
                                                       self.goal_positions, self.velocity_limits):
            if god_map.world.is_joint_continuous(name):
                error = cas.shortest_angular_distance(current, goal)
            else:
                error = goal - current

            self.add_equality_constraint(name=name,
                                         reference_velocity=velocity_limit,
                                         equality_bound=error,
                                         weight=self.weight,
                                         task_expression=current)
        joint_monitor = JointGoalReached(goal_state=goal_state,
                                         threshold=threshold)
        self.expression = joint_monitor.expression
