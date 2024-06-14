from __future__ import division

from geometry_msgs.msg import PointStamped, PoseStamped, QuaternionStamped
from geometry_msgs.msg import Vector3Stamped
from giskardpy import casadi_wrapper as cas
from giskardpy.goals.goal import Goal
from giskardpy.god_map import god_map
from giskardpy.symbol_manager import symbol_manager
from giskardpy.tasks.task import Task, WEIGHT_BELOW_CA, WEIGHT_ABOVE_CA, WEIGHT_COLLISION_AVOIDANCE
from giskardpy.goals.feature_functions import DistanceFeatureFunction, PointingFeatureFunction, \
    PerpendicularFeatureFunction, HeightFeatureFunction

from typing import Optional, List, Dict


class TCMPGoal(Goal):
    """
    A Task Critical Motion Pattern models the motions that are essential to successfully execute a task.
    For everyday activities of household robots it is assumed that they are always composed of an approach movement
    followed by a general movement function and a release movement.
    This is more or less in line with the Flanagan model and the movement function could also be further divided
    following the Flanagan model.
    """

    def __init__(self, tip_link: str, root_link: str, name: str = None,
                 movement_function: str = None,
                 reference_frame: str = None,
                 monitoring_list: List[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.TrueSymbol
                 ):
        self.root_link = god_map.world.search_for_link_name(root_link, None)
        self.tip_link = god_map.world.search_for_link_name(tip_link, None)
        if name is None:
            name = f'{self.__class__.__name__}/{self.root_link}/{self.tip_link}'
        super().__init__(name)

        # -------------- Approach -------------------------------------------------------
        # collect approach constraints and monitor their success
        # approach_task = self.create_and_add_task(task_name='approach task')

        # -------------- TCMP -----------------------------------------------------------
        # define the task critical movement using the movement function or constraints and monitor their success
        critical_task = self.create_and_add_task(task_name='critial task')

        def select_movement_function(function_description: str, params: Dict[str, float] = None):
            return cas.sin(symbol_manager.get_symbol(f'god_map.time') * 3) * 0.05

        function = select_movement_function(function_description=movement_function)
        root_P_tip = god_map.world.compose_fk_expression(self.root_link, self.tip_link).to_position()
        critical_task.add_equality_constraint_vector(reference_velocities=[0.3] * 3,
                                                     equality_bounds=(cas.Vector3([2, function, 0.9]) - root_P_tip)[:3],
                                                     weights=[WEIGHT_BELOW_CA] * 3,
                                                     task_expression=root_P_tip[:3],
                                                     names=['sdf', 'gdf', 'hgj'])

        # ------------- release ---------------------------------------------------------
        # collect release constraints and monitor their success
        # release_task = self.create_and_add_task(task_name="release task")

        # ------------ monitoring -------------------------------------------------------
        # monitor additional values throughout the motion execution
        # for monitoring_request in monitoring_list:
        #     pass

        self.connect_monitors_to_all_tasks(start_condition, hold_condition, end_condition)
