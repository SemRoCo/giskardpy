from typing import Optional

from geometry_msgs.msg import Vector3Stamped, PointStamped
from giskardpy import casadi_wrapper as cas
from giskardpy.goals.goal import Goal
from giskardpy.god_map import god_map
from giskardpy.tasks.task import WEIGHT_BELOW_CA
from giskardpy.utils import tfwrapper as tf


class PrePushDoor(Goal):

    def __init__(self,
                 root_link: str,
                 tip_link: str,
                 door_object: str,
                 door_height: float,
                 door_length: float,
                 door_depth: float,
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
            The objective is to push the object until desired rotation is reached
        """
        self.root = god_map.world.search_for_link_name(root_link, root_group)
        self.tip = god_map.world.search_for_link_name(tip_link, tip_group)
        self.door_object = god_map.world.search_for_link_name(door_object)
        object_joint_name = god_map.world.get_movable_parent_joint(self.door_object)
        object_joint_angle = god_map.world.state[object_joint_name].position

        tip_gripper_axis.header.frame_id = self.tip
        tip_gripper_axis.vector = tf.normalize(tip_gripper_axis.vector)
        object_V_object_rotation_axis = cas.Vector3(god_map.world.get_joint(object_joint_name).axis)

        self.tip_gripper_axis = tip_gripper_axis
        self.root_P_door_object = PointStamped()
        self.root_P_door_object.header.frame_id = self.root
        self.door_height = door_height
        self.door_length = door_length
        self.door_depth = door_depth
        self.reference_linear_velocity = reference_linear_velocity
        self.reference_angular_velocity = reference_angular_velocity
        self.weight = weight
        if name is None:
            name = f'{self.__class__.__name__}/{self.tip}/{self.door_object}'

        super().__init__(name)

        root_T_tip = god_map.world.compose_fk_expression(self.root, self.tip)
        root_T_door = god_map.world.compute_fk_pose(self.root, self.door_object)
        root_P_bottom_right = cas.Point3(root_T_door.pose.position)  # B
        root_P_bottom_left = cas.Point3(root_T_door.pose.position)  # A
        root_P_top_left = cas.Point3(root_T_door.pose.position)  # C

        root_P_bottom_left[0] += self.door_depth
        root_P_bottom_left[1] += self.door_length / 2

        root_P_bottom_right[0] += self.door_depth

        root_P_top_left[0] += self.door_depth
        root_P_top_left[1] += self.door_length / 2
        root_P_top_left[2] += self.door_height / 2

        root_Pose_tip = god_map.world.compute_fk_pose(self.root, self.tip)
        root_P_tip = cas.Point3(root_Pose_tip.pose.position)
        dist, root_P_nearest = cas.distance_point_to_plane(root_P_tip,
                                                           root_P_bottom_left,
                                                           root_P_bottom_right,
                                                           root_P_top_left)

        door_T_root = god_map.world.compute_fk_pose(self.door_object, self.root)
        door_P_nearest = cas.dot(cas.TransMatrix(door_T_root), root_P_nearest)

        door_rotated_R_door = cas.RotationMatrix.from_axis_angle(object_V_object_rotation_axis,
                                                                 object_joint_angle)
        door_rotated_P_nearest = cas.dot(cas.TransMatrix(door_rotated_R_door), door_P_nearest)
        root_P_nearest_in_rotated_door = cas.dot(cas.TransMatrix(root_T_door), cas.Point3(door_rotated_P_nearest))

        god_map.debug_expression_manager.add_debug_expression('goal_point_on_plane',
                                                              cas.Point3(root_P_nearest_in_rotated_door))
        god_map.debug_expression_manager.add_debug_expression('A', cas.Point3(root_P_bottom_left))
        god_map.debug_expression_manager.add_debug_expression('B', cas.Point3(root_P_bottom_right))
        god_map.debug_expression_manager.add_debug_expression('C', cas.Point3(root_P_top_left))

        push_door_task = self.create_and_add_task('pre push door')
        push_door_task.add_point_goal_constraints(frame_P_current=root_T_tip.to_position(),
                                                  frame_P_goal=cas.Point3(root_P_nearest_in_rotated_door),
                                                  reference_velocity=self.reference_linear_velocity,
                                                  weight=self.weight)

        self.connect_monitors_to_all_tasks(start_condition, hold_condition, end_condition)
