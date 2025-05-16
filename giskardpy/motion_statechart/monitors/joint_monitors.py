from typing import Dict, Optional

import giskardpy.casadi_wrapper as cas
from giskardpy.data_types.exceptions import GoalInitalizationException
from giskardpy.motion_statechart.monitors.monitors import Monitor
from giskardpy.god_map import god_map
from giskardpy.data_types.data_types import Derivatives, PrefixName


class JointGoalReached(Monitor):
    def __init__(self,
                 goal_state: Dict[str, float],
                 threshold: float = 0.01,
                 name: Optional[str] = None):
        comparison_list = []
        for joint_name, goal in goal_state.items():
            joint_name = god_map.world.search_for_joint_name(joint_name)
            current = god_map.world.get_one_dof_joint_symbol(joint_name, Derivatives.position)
            if god_map.world.is_joint_continuous(joint_name):
                error = cas.shortest_angular_distance(current, goal)
            else:
                error = goal - current
            comparison_list.append(cas.less(cas.abs(error), threshold))
        expression = cas.logic_all(cas.Expression(comparison_list))
        super().__init__(name=name)
        self.observation_expression = expression


class JointPositionAbove(Monitor):
    def __init__(self,
                 joint_name: PrefixName,
                 threshold: float,
                 name: Optional[str] = None):
        super().__init__(name=name)
        current = god_map.world.get_one_dof_joint_symbol(joint_name, Derivatives.position)
        if god_map.world.is_joint_continuous(joint_name):
            raise GoalInitalizationException(f'{self.__class__.__name__} does not support joints of type continuous.')
        expression = cas.greater(current, threshold)
        self.observation_expression = expression
