from dataclasses import dataclass

import numpy as np

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.motion_statechart.exceptions import GoalInitalizationException
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.graph_node import Goal
from giskardpy.motion_statechart.graph_node import Task
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class AlignToPushDoor(Goal):
    root_link: Body
    tip_link: Body
    door_object: Body
    door_handle: Body
    tip_gripper_axis: cas.Vector3
    reference_linear_velocity: float = 0.1
    reference_angular_velocity: float = 0.5
    weight: float = DefaultWeights.WEIGHT_BELOW_CA

    def __post_init__(self):
        """
        The objective is to reach an intermediate point before pushing the door
        """
        self.root = self.root_link
        self.tip = self.tip_link
        self.handle = self.door_handle

        object_joint_name = context.world.get_movable_parent_joint(self.door_object)
        object_joint_angle = context.world.state[object_joint_name].position

        self.tip_gripper_axis.scale(1)
        object_V_object_rotation_axis = cas.Vector3(
            context.world.get_joint(object_joint_name).axis
        )
        joint_limit = context.world.compute_joint_limits(object_joint_name, 0)

        root_T_tip = context.world._forward_kinematic_manager.compose_expression(
            self.root, self.tip
        )
        root_T_door_expr = context.world._forward_kinematic_manager.compose_expression(
            self.root, self.door_object
        )
        tip_V_tip_grasp_axis = cas.Vector3.from_iterable(self.tip_gripper_axis)
        root_V_object_rotation_axis = root_T_door_expr @ object_V_object_rotation_axis
        root_V_tip_grasp_axis = root_T_tip @ tip_V_tip_grasp_axis
        door_P_handle = context.world.compute_fk_point(self.door_object, self.handle)
        temp_point = np.asarray(
            [door_P_handle.x.to_np(), door_P_handle.y.to_np(), door_P_handle.z.to_np()]
        )
        door_P_intermediate_point = np.zeros(3)
        # axis pointing in the direction of handle frame from door joint frame
        direction_axis = np.argmax(abs(temp_point))
        door_P_intermediate_point[direction_axis] = temp_point[direction_axis] * 3 / 4
        door_P_intermediate_point = cas.Point3(
            [
                door_P_intermediate_point[0],
                door_P_intermediate_point[1],
                door_P_intermediate_point[2],
            ]
        )

        # # point w.r.t door
        desired_angle = object_joint_angle * 0.5  # just chose 1/2 of the goal angle

        # find point w.r.t rotated door in local frame
        door_R_door_rotated = cas.RotationMatrix.from_axis_angle(
            axis=object_V_object_rotation_axis, angle=desired_angle
        )
        door_T_door_rotated = cas.TransformationMatrix(door_R_door_rotated)
        # as the root_T_door is already pointing to a completely rotated door, we invert desired angle to get to the
        # intermediate point
        door_rotated_P_top = door_T_door_rotated.inverse() @ door_P_intermediate_point

        root_P_top = cas.TransformationMatrix(root_T_door_expr) @ door_rotated_P_top

        minimum_angle_to_push_door = joint_limit[1] / 4

        if object_joint_angle >= minimum_angle_to_push_door:
            context.context.add_debug_expression(
                "goal_point", root_P_top, color=Color(0, 0.5, 0.5, 1)
            )

            context.context.add_debug_expression(
                "root_V_grasp_axis", root_V_tip_grasp_axis
            )
            context.context.add_debug_expression(
                "root_V_object_axis", root_V_object_rotation_axis
            )
            align_to_push_task = Task(name="align_to_push_door")
            self.add_task(align_to_push_task)
            align_to_push_task.add_point_goal_constraints(
                frame_P_current=root_T_tip.to_position(),
                frame_P_goal=root_P_top,
                reference_velocity=self.reference_linear_velocity,
                weight=self.weight,
            )

            align_to_push_task.add_vector_goal_constraints(
                frame_V_current=root_V_tip_grasp_axis,
                frame_V_goal=root_V_object_rotation_axis,
                reference_velocity=self.reference_angular_velocity,
                weight=self.weight,
            )

        else:
            raise GoalInitalizationException(
                f"Goal cant be initialized. Failed to initialise {self.__class__.__name__}"
                "goal as the door is not open"
            )
