from typing import Optional

from giskardpy import casadi_wrapper as cas
from giskardpy.goals.goal import Goal
from giskardpy.god_map import god_map
from giskardpy.tasks.task import WEIGHT_BELOW_CA

import numpy as np


class PrePushDoor(Goal):

    def __init__(self,
                 root_link: str,
                 tip_link: str,
                 door_object: str,
                 door_handle: str,
                 root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 reference_linear_velocity: float = 0.1,
                 reference_angular_velocity: float = 0.5,
                 weight: float = WEIGHT_BELOW_CA,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol):
        """
            The objective is to push the object until desired rotation is reached
        """
        self.root = god_map.world.search_for_link_name(root_link, root_group)
        self.tip = god_map.world.search_for_link_name(tip_link, tip_group)
        self.door_object = god_map.world.search_for_link_name(door_object)
        object_joint_name = god_map.world.get_movable_parent_joint(self.door_object)
        object_V_object_rotation_axis = cas.Vector3(god_map.world.get_joint(object_joint_name).axis)

        self.handle = god_map.world.search_for_link_name(door_handle)
        self.reference_linear_velocity = reference_linear_velocity
        self.reference_angular_velocity = reference_angular_velocity
        self.weight = weight
        if name is None:
            name = f'{self.__class__.__name__}/{self.tip}/{self.door_object}'

        super().__init__(name)

        root_T_tip = god_map.world.compose_fk_expression(self.root, self.tip)
        root_T_door = god_map.world.compose_fk_expression(self.root, self.door_object)
        door_P_handle = god_map.world.compute_fk_pose(self.door_object, self.handle).pose.position
        temp_point = np.asarray([door_P_handle.x, door_P_handle.y, door_P_handle.z])

        door_V_v1 = np.zeros(3)
        # axis pointing in the direction of handle frame from door joint frame
        direction_axis = np.argmax(abs(temp_point))
        door_V_v1[direction_axis] = 1
        door_V_v2 = object_V_object_rotation_axis  # B
        door_V_v1 = cas.Vector3(door_V_v1)  # A

        door_Pose_tip = god_map.world.compose_fk_expression(self.door_object, self.tip)
        door_P_tip = door_Pose_tip.to_position()
        dist, door_P_nearest = cas.distance_point_to_plane(door_P_tip,
                                                           door_V_v2,
                                                           door_V_v1)

        root_P_nearest_in_rotated_door = cas.dot(cas.TransMatrix(root_T_door), cas.Point3(door_P_nearest))

        god_map.debug_expression_manager.add_debug_expression('goal_point_on_plane',
                                                              cas.Point3(root_P_nearest_in_rotated_door))

        push_door_task = self.create_and_add_task('pre push door')
        push_door_task.add_point_goal_constraints(frame_P_current=root_T_tip.to_position(),
                                                  frame_P_goal=cas.Point3(root_P_nearest_in_rotated_door),
                                                  reference_velocity=self.reference_linear_velocity,
                                                  weight=self.weight)

        self.connect_monitors_to_all_tasks(start_condition, hold_condition, end_condition)
