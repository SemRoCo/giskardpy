from typing import Optional
import numpy as np

from geometry_msgs.msg import Vector3Stamped, PointStamped, Quaternion
from std_msgs.msg import ColorRGBA

from giskardpy import casadi_wrapper as cas
from giskardpy.goals.goal import Goal
from giskardpy.god_map import god_map
from giskardpy.tasks.task import WEIGHT_BELOW_CA
from giskardpy.utils import tfwrapper as tf
from giskardpy.exceptions import GoalInitalizationException


class AlignToPushDoor(Goal):

    def __init__(self,
                 root_link: str,
                 tip_link: str,
                 door_object: str,
                 door_height: float,
                 door_length: float,
                 tip_gripper_axis: Vector3Stamped,
                 root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 reference_linear_velocity: float = 0.1,
                 reference_angular_velocity: float = 0.5,
                 weight: float = WEIGHT_BELOW_CA,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.TrueSymbol):
        """
        The objective is to reach an intermediate point before pushing the door
        """
        # ToDo: How to get the joint axis
        self.root = god_map.world.search_for_link_name(root_link, root_group)
        self.tip = god_map.world.search_for_link_name(tip_link, tip_group)
        self.door_object = god_map.world.search_for_link_name(door_object)
        self.door_length = door_length
        self.door_height = door_height
        self.reference_linear_velocity = reference_linear_velocity
        self.reference_angular_velocity = reference_angular_velocity
        self.weight = weight

        object_joint_name = god_map.world.get_movable_parent_joint(self.door_object)
        object_joint_angle = god_map.world.state[object_joint_name].position

        tip_gripper_axis.header.frame_id = self.tip
        tip_gripper_axis.vector = tf.normalize(tip_gripper_axis.vector)
        self.tip_gripper_axis = tip_gripper_axis
        object_V_object_rotation_axis = cas.Vector3(god_map.world.get_joint(object_joint_name).axis)

        if name is None:
            name = f'{self.__class__.__name__}/{self.root}/{self.tip}'
        super().__init__(name=name)

        root_T_tip = god_map.world.compose_fk_expression(self.root, self.tip)
        root_T_door_expr = god_map.world.compose_fk_expression(self.root, self.door_object)
        tip_V_tip_grasp_axis = cas.Vector3(self.tip_gripper_axis)
        root_V_object_rotation_axis = cas.dot(root_T_door_expr, object_V_object_rotation_axis)
        root_V_tip_grasp_axis = cas.dot(root_T_tip, tip_V_tip_grasp_axis)
        root_T_door = god_map.world.compute_fk_pose(self.root, self.door_object)
        door_T_root = god_map.world.compute_fk_pose(self.door_object, self.root)

        root_P_intermediate_point = cas.Point3(root_T_door.pose.position)
        if root_V_object_rotation_axis.y == 1:
            # 3/4 of the height as the tip has to be a little further inside the object
            root_P_intermediate_point[2] = root_P_intermediate_point[2] + self.door_height*3/4
        elif root_V_object_rotation_axis.z == 1:
            root_P_intermediate_point[1] = root_P_intermediate_point[1] + self.door_length*3/4

        # point w.r.t door
        door_P_intermediate_point = cas.dot(cas.TransMatrix(door_T_root), root_P_intermediate_point)
        desired_angle = object_joint_angle * 0.5  # just chose 1/2 of the goal angle

        # find point w.r.t rotated door in local frame
        door_rotated_R_door = cas.RotationMatrix.from_axis_angle(axis=object_V_object_rotation_axis,
                                                                 angle=desired_angle)
        door_rotated_T_door = cas.TransMatrix(door_rotated_R_door)
        door_rotated_P_top = cas.dot(door_rotated_T_door, door_P_intermediate_point)
        root_P_top = cas.dot(cas.TransMatrix(root_T_door_expr), door_rotated_P_top)

        minimum_angle_to_push_door = 0.349

        if object_joint_angle >= minimum_angle_to_push_door:
            god_map.debug_expression_manager.add_debug_expression('goal_point', root_P_top,
                                                                  color=ColorRGBA(0, 0.5, 0.5, 1))

            god_map.debug_expression_manager.add_debug_expression('root_V_grasp_axis', root_V_tip_grasp_axis)
            god_map.debug_expression_manager.add_debug_expression('root_V_object_axis', root_V_object_rotation_axis)
            align_to_push_task = self.create_and_add_task('align_to_push_door')
            align_to_push_task.add_point_goal_constraints(frame_P_current=root_T_tip.to_position(),
                                                          frame_P_goal=root_P_top,
                                                          reference_velocity=self.reference_linear_velocity,
                                                          weight=self.weight)

            align_to_push_task.add_vector_goal_constraints(frame_V_current=root_V_tip_grasp_axis,
                                                           frame_V_goal=root_V_object_rotation_axis,
                                                           reference_velocity=self.reference_angular_velocity,
                                                           weight=self.weight)

            self.connect_monitors_to_all_tasks(start_condition, hold_condition, end_condition)
        else:
            raise GoalInitalizationException(f'Goal cant be initialized. Failed to initialise {self.__class__.__name__}'
                                             'goal as the door is not open')
