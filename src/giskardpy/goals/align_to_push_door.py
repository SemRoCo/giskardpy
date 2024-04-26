from typing import Optional
import numpy as np

from geometry_msgs.msg import Vector3Stamped, PointStamped, Quaternion
from std_msgs.msg import ColorRGBA

from giskardpy import casadi_wrapper as cas
from giskardpy.goals.goal import Goal
from giskardpy.god_map import god_map
from giskardpy.symbol_manager import symbol_manager
from giskardpy.tasks.task import WEIGHT_BELOW_CA
from giskardpy.utils import tfwrapper as tf
from giskardpy.exceptions import GoalInitalizationException


class AlignToPushDoor(Goal):

    def __init__(self,
                 root_link: str,
                 tip_link: str,
                 door_object: str,
                 door_handle: str,
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
        self.root = god_map.world.search_for_link_name(root_link, root_group)
        self.tip = god_map.world.search_for_link_name(tip_link, tip_group)
        self.handle = god_map.world.search_for_link_name(door_handle)
        self.door_object = god_map.world.search_for_link_name(door_object)
        self.reference_linear_velocity = reference_linear_velocity
        self.reference_angular_velocity = reference_angular_velocity
        self.weight = weight

        object_joint_name = god_map.world.get_movable_parent_joint(self.door_object)
        object_joint_angle = god_map.world.state[object_joint_name].position

        tip_gripper_axis.header.frame_id = self.tip
        tip_gripper_axis.vector = tf.normalize(tip_gripper_axis.vector)
        self.tip_gripper_axis = tip_gripper_axis
        object_V_object_rotation_axis = cas.Vector3(god_map.world.get_joint(object_joint_name).axis)
        joint_limit = god_map.world.compute_joint_limits(object_joint_name, 0)

        if name is None:
            name = f'{self.__class__.__name__}/{self.root}/{self.tip}'
        super().__init__(name=name)

        root_T_tip = god_map.world.compose_fk_expression(self.root, self.tip)
        root_T_door_expr = god_map.world.compose_fk_expression(self.root, self.door_object)
        tip_V_tip_grasp_axis = cas.Vector3(self.tip_gripper_axis)
        root_V_object_rotation_axis = cas.dot(root_T_door_expr, object_V_object_rotation_axis)
        root_V_tip_grasp_axis = cas.dot(root_T_tip, tip_V_tip_grasp_axis)
        door_P_handle = god_map.world.compute_fk_pose(self.door_object, self.handle).pose.position
        temp_point = np.asarray([door_P_handle.x, door_P_handle.y, door_P_handle.z])
        door_P_intermediate_point = np.zeros(3)
        # axis pointing in the direction of handle frame from door joint frame
        direction_axis = np.argmax(abs(temp_point))
        door_P_intermediate_point[direction_axis] = temp_point[direction_axis]*3/4
        door_P_intermediate_point = cas.Point3([door_P_intermediate_point[0],
                                                door_P_intermediate_point[1],
                                                door_P_intermediate_point[2]])

        # # point w.r.t door
        desired_angle = object_joint_angle * 0.5  # just chose 1/2 of the goal angle

        # find point w.r.t rotated door in local frame
        door_R_door_rotated = cas.RotationMatrix.from_axis_angle(axis=object_V_object_rotation_axis,
                                                                 angle=desired_angle)
        door_T_door_rotated = cas.TransMatrix(door_R_door_rotated)
        # as the root_T_door is already pointing to a completely rotated door, we invert desired angle to get to the
        # intermediate point
        door_rotated_P_top = cas.dot(door_T_door_rotated.inverse(), door_P_intermediate_point)
        root_P_top = cas.dot(cas.TransMatrix(root_T_door_expr), door_rotated_P_top)

        minimum_angle_to_push_door = joint_limit[1]/4

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
