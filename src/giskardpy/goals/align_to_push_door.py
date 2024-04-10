from typing import Optional
import numpy as np

from geometry_msgs.msg import Vector3Stamped, PointStamped, Quaternion
from std_msgs.msg import ColorRGBA

from giskardpy import casadi_wrapper as cas
from giskardpy.goals.goal import Goal
from giskardpy.god_map import god_map
from giskardpy.monitors.monitors import ExpressionMonitor
from giskardpy.tasks.task import WEIGHT_BELOW_CA
from giskardpy.utils import tfwrapper as tf, logging
from giskardpy.symbol_manager import symbol_manager
from giskardpy.exceptions import GoalInitalizationException


class AlignToPushDoor(Goal):
    # start condition - door is open a little bit
    # hold condition - door is open a little bit
    # end condition - gripper is in the gap between the open door and the dishwasher
    def __init__(self,
                 root_link: str,
                 tip_link: str,
                 door_object: str,
                 # you will need the intermediate point in the direction the object opens. In dishwasher along z axis
                 # in door along y axis?
                 door_height: float,
                 door_length: float,
                 object_joint_name: str,
                 tip_gripper_axis: Vector3Stamped,
                 object_rotation_axis: Vector3Stamped,
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
        self.door_object = god_map.world.search_for_link_name(door_object)
        self.object_joint_angle = \
            god_map.world.state.to_position_dict()[god_map.world.search_for_joint_name(object_joint_name)]

        tip_gripper_axis.header.frame_id = self.tip
        tip_gripper_axis.vector = tf.normalize(tip_gripper_axis.vector)
        object_rotation_axis.header.frame_id = self.door_object
        object_rotation_axis.vector = tf.normalize(object_rotation_axis.vector)

        self.tip_gripper_axis = tip_gripper_axis
        self.object_rotation_axis = object_rotation_axis
        self.root_P_door_object = PointStamped()
        self.root_P_door_object.header.frame_id = self.root

        # Pose of the articulated object is at the center of the axis along which it rotates and not at the center of
        # the object.
        self.door_length = door_length
        self.door_height = door_height
        self.reference_linear_velocity = reference_linear_velocity
        self.reference_angular_velocity = reference_angular_velocity
        self.weight = weight
        if name is None:
            name = f'{self.__class__.__name__}/{self.root}/{self.tip}'
        super().__init__(name=name)

        root_T_tip = god_map.world.compose_fk_expression(self.root, self.tip)
        root_T_door_expr = god_map.world.compose_fk_expression(self.root, self.door_object)
        object_V_object_rotation_axis = cas.Vector3(self.object_rotation_axis)
        tip_V_tip_grasp_axis = cas.Vector3(self.tip_gripper_axis)
        root_V_object_rotation_axis = cas.dot(root_T_door_expr, object_V_object_rotation_axis)
        root_V_tip_grasp_axis = cas.dot(root_T_tip, tip_V_tip_grasp_axis)
        root_T_door = god_map.world.compute_fk_pose(self.root, self.door_object)
        door_T_root = god_map.world.compute_fk_pose(self.door_object, self.root)

        root_P_intermediate_point = PointStamped()
        root_P_intermediate_point.header.frame_id = self.root
        root_P_intermediate_point.point.x = root_T_door.pose.position.x
        root_P_intermediate_point.point.y = root_T_door.pose.position.y
        root_P_intermediate_point.point.z = root_T_door.pose.position.z
        if root_V_object_rotation_axis.y == 1:
            root_P_intermediate_point.point.z = root_P_intermediate_point.point.z + self.door_height
        elif root_V_object_rotation_axis.z == 1:
            root_P_intermediate_point.point.y = root_P_intermediate_point.point.y + self.door_length

        # point w.r.t door
        door_P_intermediate_point = cas.dot(cas.TransMatrix(door_T_root), cas.Point3(root_P_intermediate_point))
        desired_angle = self.object_joint_angle * 0.5  # just chose 1/2 of the goal angle

        # find point w.r.t rotated door in local frame
        door_rotated_R_door = cas.RotationMatrix.from_axis_angle(axis=object_V_object_rotation_axis,
                                                                 angle=desired_angle)
        door_rotated_T_door = cas.TransMatrix(door_rotated_R_door)
        door_rotated_P_top = cas.dot(door_rotated_T_door, cas.Point3(door_P_intermediate_point))
        root_P_top = cas.dot(cas.TransMatrix(root_T_door_expr), cas.Point3(door_rotated_P_top))

        minimum_angle_to_push_door = 0.349
        # door_open = self.object_joint_angle >= minimum_angle_to_push_door
        door_open = cas.greater_equal(self.object_joint_angle, minimum_angle_to_push_door)

        try:
            if symbol_manager.evaluate_expr(expr=door_open):
                god_map.debug_expression_manager.add_debug_expression('goal_point', cas.Point3(root_P_top),
                                                                      color=ColorRGBA(0, 0.5, 0.5, 1))

                god_map.debug_expression_manager.add_debug_expression('root_V_grasp_axis', root_V_tip_grasp_axis)
                god_map.debug_expression_manager.add_debug_expression('root_V_object_axis', root_V_object_rotation_axis)
                align_to_push_task = self.create_and_add_task('align_to_push_door')
                align_to_push_task.add_point_goal_constraints(frame_P_current=root_T_tip.to_position(),
                                                              frame_P_goal=cas.Point3(root_P_top),
                                                              reference_velocity=self.reference_linear_velocity,
                                                              weight=self.weight)

                align_to_push_task.add_vector_goal_constraints(frame_V_current=cas.Vector3(root_V_tip_grasp_axis),
                                                               frame_V_goal=cas.Vector3(root_V_object_rotation_axis),
                                                               reference_velocity=self.reference_angular_velocity,
                                                               weight=self.weight)

                # align_to_push_task.start_condition = door_opened_monitor
                # align_to_push_task.end_condition = tip_aligned_monitor
                self.connect_monitors_to_all_tasks(start_condition, hold_condition, end_condition)
            else:
                raise GoalInitalizationException('Goal cant be initialized')

        except GoalInitalizationException as e:
            logging.loginfo(f'{str(e)}.Failed to initialise {self.__class__.__name__} goal as '
                            f'the door is not open')

        # door_opened_monitor = ExpressionMonitor(name='door opened', stay_true=True, start_condition=start_condition)
        # self.add_monitor(door_opened_monitor)
        # door_opened_monitor.expression = door_open


